import imageio
# from pygifsicle import optimize
import argparse
import os
import random
import shutil
from datetime import datetime
import numpy as np
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model_hypoconv import HypoConvnets, VideoDataSet, adjust_lr, worker_init_fn, make_coord_grid, all_reduce
from torchvision.utils import save_image
from pytorch_msssim import ssim
from lpips import LPIPS
from pandas import DataFrame

def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--vid_path', type=str, default='data/k400/1S0f3iBwy4o_000004_000014.mp4', help='video path',)
    parser.add_argument('--vid', type=str, default='k400_1', help='video path',)
    parser.add_argument('--frames', type=int, default=4, help='video frame number',)
    parser.add_argument('--crop_size', type=int, default=128, help='video crop size',)
    parser.add_argument('--hid_dim', type=int, default=32, help='hid_dim for hypo_convnets',)
    parser.add_argument('--strds', type=int, nargs='+', default=[8,4,2,2], help='strds list',)
    parser.add_argument('--ks', type=int, nargs='+', default=[1,3], help='kernel size list',)
    parser.add_argument('--pe_dim', type=int, default=48, help='pe_dim')
    parser.add_argument('--pe_sigma', type=int, default=1024, help='pe_dim')
    parser.add_argument('--act', type=str, default='gelu', help='act type')
    parser.add_argument('--out_bias', type=str, default='tanh', help='out_bias type')

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default='150', help='epoch num')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate, default=0.0005')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--dump_videos', action='store_true', default=False, help='concat the prediction images into video')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    args = parser.parse_args()
    torch.set_printoptions(precision=4) 
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    exp_id = f'{args.vid}/e{args.epochs}_b{args.batchSize}_lr{args.lr}_{args.lr_type}{args.suffix}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)


def data_to_gpu(x, device):
    return x.to(device)


def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)

    args.dump_vis = (args.dump_images or args.dump_videos)

    # setup dataloader    
    full_dataset = VideoDataSet(args)
    args.frames_num = len(full_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(full_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)
    val_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batchSize,  shuffle=False,
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False, worker_init_fn=worker_init_fn)

    # Building model
    model = HypoConvnets(args)

    ##### get model params and flops #####
    if local_rank in [0, None]:
        total_params = sum(p.numel() for p in model.parameters())
        param_str = f'Total_{total_params/1e6}M'
        print(f'{args}\n {model}\n {param_str}', flush=True)
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'{param_str}\n')
        writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model)
    elif torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=0.)

    # resume from args.weight
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt, strict=False)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt, strict=False)
        else:
            model.load_state_dict(new_ckt, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    if args.start_epoch < 0:
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] 
        args.start_epoch = max(args.start_epoch, 0)

    # Training
    start = datetime.now()
    psnr_list, best_eval_psnr = [], 0
    total_epochs = args.epochs

    if args.eval_only:
        print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight)
        eval_out = evaluate(model, val_dataloader, local_rank, args, args.dump_vis)
        print_str = f'PSNR for output {hw}: {round(eval_psnr, 2)}'
        if local_rank in [0, None]:
            print(print_str, flush=True)
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n\n')        
            args.train_time, args.cur_epoch = 0, args.epochs
            Dump2CSV(args, best_eval_psnr, eval_out, [], 'eval.csv')
        return

    for epoch in range(args.start_epoch, total_epochs):
        model.train() 
        epoch_start_time = datetime.now()
        loss_list, pred_psnr_list = [], []
        # iterate over dataloader
        device = next(model.parameters()).device
        coord = make_coord_grid((args.frames_num,), (-1, 1), device=device)
        for i, sample in enumerate(train_dataloader):
            img_data, img_idx = data_to_gpu(sample['frame'], device), data_to_gpu(sample['idx'], device)
            img_idx = coord.index_select(0,img_idx)
            if i > 10 and args.debug:
                break
            cur_epoch = (epoch + float(i)) / len(train_dataloader) / total_epochs
            lr = adjust_lr(optimizer, cur_epoch, args)
            img_out = model(img_idx)
            # save hypo_convnet checkpoint to fast loading
            # torch.jit.save(torch.jit.trace(model, img_idx), f'hypo_convnet.pth')
            
            final_loss = F.mse_loss(img_out, img_data)      
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            loss_list.append(final_loss.detach())
            if i+1 % args.print_freq == 0 or i == len(train_dataloader) - 1:
                avg_loss = sum(loss_list) / len(loss_list)
                pred_psnr = -10 * torch.log10(avg_loss).item()
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e},  pred_PSNR: {}'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, total_epochs, i+1, len(train_dataloader), lr, 
                     round(pred_psnr, 2))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            pred_psnr = all_reduce([pred_psnr.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, None]:
            h, w = img_out.shape[-2:]
            writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch+1)
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or (total_epochs - epoch) in [5, 2]:
            # eval_psnr, hw 
            eval_out = evaluate(model, val_dataloader, local_rank, args, 
                args.dump_vis if epoch == total_epochs - 1 else False)
            eval_psnr, hw = eval_out['psnr'], eval_out['hw']
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                print_str = f'Eval at epoch {epoch+1} for {hw}: '
                best_eval_psnr = eval_psnr if eval_psnr > best_eval_psnr else best_eval_psnr
                writer.add_scalar(f'Val/psnr_{hw}', eval_psnr, epoch+1)
                writer.add_scalar(f'Val/best_psnr_{hw}', best_eval_psnr, epoch+1)
                print_str += f'psnr: {round(eval_psnr, 2)} | {round(best_eval_psnr, 2)} '
                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
            psnr_list.append(eval_psnr)

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),   
        }    
        if local_rank in [0, None]:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if (epoch + 1) % total_epochs == 0:
                args.cur_epoch = epoch + 1
                args.train_time = str(datetime.now() - start)
                Dump2CSV(args, best_eval_psnr, eval_out, psnr_list, f'epoch{epoch+1}.csv')
                torch.save(save_checkpoint, f'{args.outf}/epoch{epoch+1}.pth')

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")


# Writing final results in CSV file
def Dump2CSV(args, best_psnr, output, psnr_list, filename='results.csv'):
    # import pdb; pdb.set_trace; from IPython import embed; embed() 
    csv_dict = {'Dataset': args.vid, 'CurEpoch': args.cur_epoch, 'Time': args.train_time, 
                'pe_dim': args.pe_dim, 'pe_sigma': args.pe_sigma, 'Frames': args.frames, 
                'CropSize': args.crop_size, 'Strds': args.strds, 'Hid_dim': args.hid_dim, 'KS': args.ks,
                'Lr_type': args.lr_type, 'Lr (E-3)': args.lr*1e3, 'Batch': args.batchSize,
                'Epoch': args.epochs, 'Act': args.act,  f'PSNR_list_{args.eval_freq}': ','.join([str(round(v, 2)) for v in psnr_list]),
                'best_psnr': best_psnr, 'eval_psnr': output['psnr'], 'eval_ssim': output['ssim'], 
                'eval_lpips': output['lpips']
        }   
    csv_df = DataFrame.from_dict(csv_dict, orient='index').T
    csv_df.to_csv(os.path.join(args.outf, filename))


from pytorch_msssim import ssim
@torch.no_grad()
def evaluate(model, full_dataloader, local_rank, args, dump_vis=False):
    model.eval()
    device = next(model.parameters()).device
    coord = make_coord_grid((args.frames_num,), (-1, 1), device=device)
    psnr_list, ssim_list, lpips_list = [], [], []
    lpips_fn = LPIPS(net='alex').to('cuda')
    lpips_fn.eval()
    for i, sample in enumerate(full_dataloader):
        img_data, img_idx = data_to_gpu(sample['frame'], device), data_to_gpu(sample['idx'], device)
        img_idx = coord.index_select(0,img_idx)
        if i > 10 and args.debug:
            break
        img_out = model(img_idx)
        mse_loss = F.mse_loss(img_out, img_data, reduction='none').mean((1,2,3)).detach()
        # pred_psnr = ssim(img_out.detach(), img_data, data_range=1, size_average=False).cpu()
        pred_psnr = -10 * torch.log10(mse_loss).cpu()

        ssim_v = ssim(img_out.detach(), img_data.detach(), data_range=1).cpu()
        lpips_v = lpips_fn((img_out.detach()-0.5)*2, (img_data.detach()-0.5)*2 ).mean().cpu()
        psnr_list.append(pred_psnr)
        ssim_list.append(ssim_v)
        lpips_list.append(lpips_v)

        # dump predictions
        if dump_vis:
            visual_dir = f'{args.outf}/visualize'
            print(f'Saving predictions to {visual_dir}...')
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)            
            for batch_ind, cur_img_idx in enumerate(img_idx):
                full_ind = i * args.batchSize + batch_ind
                dump_img_list =  [img_data[batch_ind], img_out[batch_ind]]                
                concat_img = torch.cat(dump_img_list, dim=2)    #img_out[batch_ind], 
                save_image(concat_img, f'{visual_dir}/pred_{full_ind:04d}_{pred_psnr[batch_ind]}.png')

        if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
            # import pdb; pdb.set_trace; from IPython import embed; embed()     
            avg_psnr = torch.cat(psnr_list).mean().item()
            avg_ssim = torch.stack(ssim_list).mean().item()
            avg_lpips = torch.stack(lpips_list).mean().item()
            print_str = '[{}] Rank:{}, Eval [{}/{}], psnr: {}'.format(datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                local_rank, i+1, len(full_dataloader), round(avg_psnr, 2) )
            if local_rank in [0, None]:
                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')

        h,w = img_data.shape[-2:]
        if args.distributed and args.ngpus_per_node > 1:
            pred_psnr = all_reduce([pred_psnr.to(local_rank)])

        if dump_vis and args.dump_videos:
            gif_file = f'{args.outf}/source_target.gif'
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in sorted(os.listdir(visual_dir)):
                    image = imageio.imread(os.path.join(visual_dir, filename))
                    writer.append_data(image)
            if not args.dump_images:
                shutil.rmtree(visual_dir)
            # optimize(gif_file)
    out_dict = {'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips, 'hw': (h,w)}
    return out_dict
# avg_psnr, (h,w)


if __name__ == '__main__':
    main()
