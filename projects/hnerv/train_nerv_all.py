# from pygifsicle import optimize
import argparse
import os
import random
import shutil
from datetime import datetime

import imageio
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from hnerv_utils import *
from model_all import HNeRV, HNeRVDecoder, TransformInput, VideoDataSet, PPrint


def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='', help='data path for vid')
    parser.add_argument('--vid', type=str, default='k400_train0', help='video id',)
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='video crop size',)
    parser.add_argument('--resize_list', type=str, default='-1', help='video resize size',)

    # NERV architecture parameters
    # Embedding and encoding parameters
    parser.add_argument('--embed', type=str, default='', help='empty string for HNeRV, and base value/embed_length for NeRV position encoding')
    parser.add_argument('--ks', type=str, default='0_3_3', help='kernel size for encoder and decoder')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[], help='stride list for encoder')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='enc latent dim and embedding ratio')
    parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
    parser.add_argument('--interpolate', type=str, default='none', help='interpolation mode', choices=['none', 't', 'hw', 'thw'])

    # Decoding parameters: FC + Conv
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
    parser.add_argument('--reduce', type=float, default=1.2, help='chanel reduction for next stage')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list for decoder')
    parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel'], type=str, nargs="+",
        help='conv type for encoder/decoder', )
        # choices=['pshuffel', 'conv', 'convnext', 'interpolate'])
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', 
                    choices=['none', 'bn', 'in', 'ln'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', 
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Epoch number')
    parser.add_argument('--block_params', type=str, default='1_1', help='residual blocks and percentile to save')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
    parser.add_argument('--loss', type=str, default='Fusion6', help='loss type, default=L2')
    parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for model quantization')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='bit length for embedding quantization')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--dump_videos', action='store_true', default=False, help='concat the prediction images into video')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    parser.add_argument('--encoder_file',  default='', type=str, help='specify the embedding file')

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

    args.enc_strd_str, args.dec_strd_str = ','.join([str(x) for x in args.enc_strds]), ','.join([str(x) for x in args.dec_strds])
    extra_str = 'Size{}_ENC_{}_{}_DEC_{}_{}_{}{}{}'.format(args.modelsize, args.conv_type[0], args.enc_strd_str, 
        args.conv_type[1], args.dec_strd_str, '' if args.norm == 'none' else f'_{args.norm}', 
        '_dist' if args.distributed else '', '_shuffle_data' if args.shuffle_data else '',)
    args.quant_str = f'qM{args.quant_model_bit}_E{args.quant_embed_bit}'
    embed_str = f'{args.embed}_Dim{args.enc_dim}'
    exp_id = f'{args.vid}/{args.data_split}_{embed_str}_FC{args.fc_hw}_K{args.ks}_r{args.reduce}_low{args.lower_width}_blk{args.num_blks}_{args.interpolate}' + \
            f'_e{args.epochs}_b{args.batchSize}_{args.quant_str}_lr{args.lr}_{args.lr_type}_{args.loss}_{extra_str}{args.act}{args.block_params}{args.suffix}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)
    args.log_file = f'{args.outf}/log.txt'

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

    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim',]
        # 'quant_seen_psnr', 'quant_seen_ssim', 'quant_unseen_psnr', 'quant_unseen_ssim']
    args.cur_metric_list, args.best_metric_list = [[torch.tensor(0) for _ in range(len(args.metric_names))] for __ in range(2)]

    # setup dataloader    
    full_dataset = VideoDataSet(args)
    sampler = torch.utils.data.distributed.DistributedSampler(full_dataset) if args.distributed else None
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batchSize, shuffle=(sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=False, worker_init_fn=worker_init_fn)
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)
    args.dump_vis = (args.dump_images or args.dump_videos)

    #  Make sure the testing dataset is fixed for every run
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # Compute the parameter number
    if 'pe' in args.embed or 'le' in args.embed:
        embed_param = 0
        embed_dim = int(args.embed.split('_')[-1]) * 2
        fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
    else:
        total_enc_strds = np.prod(args.enc_strds)
        embed_hw = args.final_size / total_enc_strds**2
        enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
        embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
        embed_num = 2 if 't' in args.interpolate else args.full_data_length
        embed_param = embed_num * float(embed_dim) * args.final_size / total_enc_strds**2
        args.enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
        fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

    decoder_size = args.modelsize * 1e6 - embed_param
    ch_reduce = 1. / args.reduce
    dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
    def cur_ks(i, dec_ks1=dec_ks1, dec_ks2=dec_ks2):
        return min((2*i + dec_ks1), dec_ks2)
    dec_blks = int(args.num_blks.split('_')[-1])

    if 'dense_conv' in args.conv_type[1]:
        a = sum([cur_ks(i)**2 * ch_reduce**(2*i) * ((1 if 'interpolate' in args.conv_type[1] else s)**2 + (dec_blks-1)*ch_reduce) 
                for i,s in enumerate(args.dec_strds)]) * ch_reduce
        b =  embed_dim * fc_param 
    elif 'split_conv' in args.conv_type[1]:
        a = sum([ch_reduce**(2*i) * ((1 if 'interpolate' in args.conv_type[1] else s)**2 + (dec_blks-1)*ch_reduce) 
                for i,s in enumerate(args.dec_strds)]) * ch_reduce
        b =  embed_dim * fc_param + sum([ch_reduce**(i+1) for i in range(len(args.dec_strds))])
    elif 'bottleneck' in args.conv_type[1]:
        extra_ks, extra_exp = [int(x) for x in args.conv_type[1].split('_')[-2:]]
        a = sum([(1+extra_ks**2*ch_reduce) * ch_reduce**(2*i) * ((1 if 'interpolate' in args.conv_type[1] else s)**4 + (dec_blks-1)*ch_reduce)
                for i,s in enumerate(args.dec_strds)]) * ch_reduce * extra_exp
        b =  embed_dim * fc_param + sum([ch_reduce**(i+1)*cur_ks(i)**2 for i in range(len(args.dec_strds))])

    args.fc_dim = int(np.roots([a, b, -decoder_size]).max())
    # import pdb; pdb.set_trace; from IPython import embed; embed()     

    # Building model
    model = HNeRV(args)

    ##### get model params and flops #####
    if local_rank in [0, None]:
        encoder_param = (sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6) 
        decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6) 
        total_param = decoder_param + embed_param / 1e6
        args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
        param_str = f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 2)}M_Total_{round(total_param, 2)}M'
        PPrint(str(model) + '\n' + f'{param_str}\n', local_rank, args)
        args.writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))
    else:
        args.writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model)
    elif torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=0.)
    args.transform_func = TransformInput(args)
    args.psnr_list = []
    args.start_time = datetime.now()

    # resume from args.weight
    checkpoint = None
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

    if args.eval_only:
        print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight)
        Evaluate(model, full_dataloader, args.dump_vis, local_rank, args)
        out_str_list = ['{}: {}'.format(k, RoundTensor(v, 2 if 'psnr' in k else 4)
            ) for k,v in zip(args.metric_names, args.best_metric_list)]
        PPrint(print_str + '/t'.join(out_str_list), local_rank, args)
        if local_rank in [0, None]:
            Dump2CSV(args, 'eval.csv')

        return

    # Training
    for epoch in range(args.start_epoch, args.epochs):
        Train_one_epoch(model, train_dataloader, optimizer, epoch, local_rank, args)
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            Evaluate(model, full_dataloader, (args.dump_vis and epoch==args.epochs-1), local_rank, args, epoch+1)

        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),   
        }    
        if local_rank in [0, None]:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if (epoch + 1) % args.epochs == 0:
                Dump2CSV(args)
    PPrint(f"Training complete in: {str(datetime.now() - args.start_time)}", local_rank, args)                


def Train_one_epoch(model, dataloader, optimizer, epoch, local_rank, args):
    model.train()       
    epoch_start_time = datetime.now()
    pred_psnr_list = []
    # iterate over dataloader
    device = next(model.parameters()).device
    for i, sample in enumerate(dataloader):
        cur_epoch = (epoch + float(i) / len(dataloader)) / args.epochs
        lr = adjust_lr(optimizer, cur_epoch, args)
        if i > 10 and args.debug:
            break
        # forward and backward
        sample = {k:data_to_gpu(v, device) for k,v in sample.items()}  
        out_dict = model(sample) 
        img_out, img_gt = out_dict['img_out'], sample['img']
        final_loss = loss_fn(img_out, img_gt, args.loss)
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_gt)) 
        if i % args.print_freq == 0 or i == len(dataloader) - 1:
            pred_psnr = torch.cat(pred_psnr_list).mean()
            print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {}'.format(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, args.epochs, i+1, len(dataloader), lr, 
                RoundTensor(pred_psnr, 2))
            PPrint(print_str, local_rank, args)

    epoch_time = (datetime.now()-epoch_start_time).total_seconds()
    PPrint(f'training time/epoch: {epoch_time}s', local_rank, args)

    # collect numbers from other gpus
    if args.distributed and args.ngpus_per_node > 1:
        pred_psnr = all_reduce([pred_psnr.to(local_rank)])

    # ADD train_PSNR TO TENSORBOARD
    if local_rank in [0, None]:
        h, w = img_out.shape[-2:]
        args.writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch+1)
        args.writer.add_scalar('Train/lr', lr, epoch+1)


@torch.no_grad()
def Evaluate(model, dataloader, dump_vis, local_rank, args, epoch=0):
    metric_list, time_list = [[] for _ in range(len(args.metric_names))], []
    model.eval()
    device = next(model.parameters()).device
    if dump_vis:
        visual_dir = f'{args.outf}/visualize_imgs'
        print(f'Saving predictions to {visual_dir}...')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)     

    for i, sample in enumerate(dataloader):
        if i > 10 and args.debug:
            break
        sample = {k:data_to_gpu(v, device) for k,v in sample.items()}  
        out_dict = model(sample)
        time_list.append(out_dict['dec_time'])
        img_out, img_gt, img_idx = out_dict['img_out'], sample['img'], sample['idx']

        # compute psnr and ms-ssim
        pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch([img_out], img_gt)
        for metric_idx, cur_v in  enumerate([pred_psnr, pred_ssim]):
            for batch_i, cur_img_idx in enumerate(img_idx):
                metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                metric_list[metric_idx_start+metric_idx].append(cur_v[:,batch_i])

        # dump predictions
        if dump_vis:
            for batch_ind, cur_img_idx in enumerate(img_idx):
                full_ind = i * args.batchSize + batch_ind
                dump_img_list = [img_gt[batch_ind], img_out[batch_ind]]
                temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                concat_img = torch.cat(dump_img_list, dim=2)    #img_out[batch_ind], 
                save_image(concat_img, f'{visual_dir}/pred_{full_ind:04d}_{temp_psnr_list}.png')

        # print eval results and add to log txt
        if i % args.print_freq == 0 or i == len(dataloader) - 1:
            avg_time = sum(time_list) / len(time_list)
            fps = args.batchSize / avg_time
            print_str = '[{}] Rank:{}, Eval at Step [{}/{}], FPS {}, '.format(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, i+1, len(dataloader), round(fps, 1))
            for v_name, v_list in zip(args.metric_names, metric_list):
                cur_value = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                print_str += f'{v_name}: {RoundTensor(cur_value, 2)} | '
            PPrint(print_str, local_rank, args)

        # import pdb; pdb.set_trace; from IPython import embed; embed()     
        # Collect results from other gpus
        results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
        args.fps = fps
        if args.distributed and args.ngpus_per_node > 1:
            for cur_v in results_list:
                cur_v = all_reduce([cur_v.to(local_rank)])

        # Dump predictions and concat into videos
        if dump_vis and args.dump_videos:
            gif_file = os.path.join(args.outf, 'gt_pred.gif')
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in sorted(os.listdir(visual_dir)):
                    image = imageio.v2.imread(os.path.join(visual_dir, filename))
                    writer.append_data(image)
            if not args.dump_images:
                shutil.rmtree(visual_dir)
            # optimize(gif_file)

    if local_rank in [0, None]:
        # ADD val_PSNR TO TENSORBOARD
        print_str = f'Eval: '
        for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, args.best_metric_list, results_list)):
            best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
            if 'psnr' in metric_name:
                args.writer.add_scalar(f'Val/{metric_name}', metric_value.max(), epoch)
                args.writer.add_scalar(f'Val/best_{metric_name}', best_metric_value, epoch)
                if metric_name == 'pred_seen_psnr':
                    args.psnr_list.append(metric_value.max())
                print_str += f'{metric_name}: {RoundTensor(metric_value, 2)} | '
            args.best_metric_list[i] = best_metric_value
        PPrint(print_str, local_rank, args)


# Writing final results in CSV file
def Dump2CSV(args, filename='results.csv'):
    result_dict = {'Vid':args.vid, 'Time':datetime.now()-args.start_time, 
        'FPS':args.fps, 'Split':args.data_split, 'Embed':args.embed, 'Crop': args.crop_list,
        'Resize':args.resize_list, 'Lr_type':args.lr_type, 'LR (E-3)': args.lr*1e3, 'Batch':args.batchSize,
        'Size (M)': f'{round(args.encoder_param, 2)}_{round(args.decoder_param, 2)}_{round(args.total_param, 2)}', 
        'ModelSize': args.modelsize, 'Interpolate': args.interpolate, 'Epoch':args.epochs, 'Loss':args.loss, 
        'Act':args.act, 'Norm':args.norm, 'Blks': args.num_blks,
        'FC':args.fc_hw, 'Reduce':args.reduce, 'ENC_type':args.conv_type[0], 'ENC_strds':args.enc_strd_str, 'KS':args.ks,
        'enc_dim':args.enc_dim, 'DEC':args.conv_type[1], 'DEC_strds':args.dec_strd_str, 'lower_width':args.lower_width,
         'Quant':args.quant_str, f'PSNR_list_{args.eval_freq}':','.join([RoundTensor(v, 2) for v in args.psnr_list]),}
        #  'bits/param':args.bits_per_param, 'bits/param w/ overhead':args.full_bits_per_param, 'bits/pixel':args.total_bpp,
    result_dict.update({f'best_{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, args.best_metric_list)})
    # result_dict.update({f'{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, args.results_list) if 'pred' in k})
    csv_path = os.path.join(args.outf, filename)
    print(f'results dumped to {csv_path}')
    pd.DataFrame(result_dict,index=[0]).to_csv(csv_path)


if __name__ == '__main__':
    main()
