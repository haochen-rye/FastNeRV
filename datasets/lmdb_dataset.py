"""LMBD dataset to wrap an existing dataset and "seamlessly" move all data into an LMDB."""

import os
import warnings
import time

import platform
import lmdb
from copy import deepcopy
from glob import glob
from math import ceil
import pandas as pd
import json
import shutil
from dahuffman import HuffmanCodec

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.io import write_video, read_video
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, InterpolationMode, Normalize, Resize
import cv2

from utils import quant_tensor, ModulateNeRV
from models import make as make_model

warnings.filterwarnings("ignore", "The given buffer is not writable", UserWarning)

import logging
log = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')
ANNO_FILE = 'anno.js'
NUM_ATTEMPTS = 60*24

class LMDB_config:
    """Configuration options for the LMDB. These should be reasonable defaults.

    For large datasets with random access (for example ImageNet trained with shuffle=True), try setting readahead=False.
    """

    # Writing:
    map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    write_frequency = 10240  # how often to flush during database creation
    shuffle_while_writing = False  # Shuffle during DB creation
    db_channels_first = True
    rounds = 1  # Can write multiple rounds of the same dataset (for example with different augmentations).

    # reading:
    max_readers = 128
    readahead = True  # this should be beneficial for long sequential reads, disable when randomly accessing large DBs
    meminit = True
    max_spare_txns = 128


class LmdbDataset(torch.utils.data.Dataset):
    """Implement LMDB caching and access for image-based pytorch dataset.

    Use this class to wrap an existing dataset. After wrapping, the wrapped dataset behaves like the original dataset,
    but the underlying images are moved into an LMDB (Lightning Memory-Mapped Database Manager).

    When temporary_db=False and force_db_rebuild=False, the second invocation of a database with the same name at the
    same root location will read the existing database instead of building another one. It is the user's responsibility
    to make sure that the existing database with the same name contains the same data.
    """

    def __init__(self, dataset, args, can_create=True, db_cfg=LMDB_config):
        """Initialize with given pytorch dataset, figure out transforms to apply and read or create the database."""
        self.data_len, self.num_per_shard, self.frame_num = len(dataset), int(1e4), dataset.frame_num
        root, name, rebuild, lmdb_frames, lmdb_strd = [args[x] for x in ['path', 'name', 'rebuild', 'frames', 'strd']]
        self.lmdb_crop, self.lmdb_aug = deepcopy(dataset.lmdb_crop), deepcopy(dataset.lmdb_aug)
        self.rand_flip, self.scale_resize = deepcopy(dataset.rand_flip), deepcopy(dataset.scale_resize)
        self.tfm_video, self.split = dataset.tfm_video, dataset.split
        self.crop_size, self.border_size, self.cls_vid_num = dataset.crop_size, dataset.border_size, dataset.cls_vid_num
        shape_str = f'L{len(dataset)}_T{lmdb_frames}_sample_{lmdb_strd}_Size_{dataset.crop_size}+{dataset.border_size}_ClsVidNum_{dataset.cls_vid_num}'
        out_dir = os.path.expanduser(os.path.join(root, f"{name}_{shape_str}"))
        self.anno_file_path = os.path.join(out_dir, ANNO_FILE)
        self.db_cfg = db_cfg

        # create LMDB dataset if needed
        if can_create and (not self.complete_lmdb(out_dir) or rebuild.lower() in ['yes', 'true']):
            create_dataset = True
            if os.path.isdir(out_dir):
                # new_dir = out_dir + '_old'
                # log.info(f'renaming {out_dir} to {new_dir}')
                # os.rename(out_dir, new_dir)
                log.info(f'removing {out_dir}')
                shutil.rmtree(out_dir)
        else:
            create_dataset = False

        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        if create_dataset:
            self._create_dataset(dataset, out_dir, num_per_shard=self.num_per_shard)
        else:
            for _ in range(NUM_ATTEMPTS):
                if os.path.isdir(out_dir):
                    log.info(f'Reusing cached database at {out_dir}')
                    break
                time.sleep(60)

        self.db_files = sorted(glob(os.path.join(out_dir, 'shard*.lmdb')))
        self.db_list = None
        if self.anno_file_path.endswith('csv'):
            anno_data = pd.read_csv(self.anno_file_path)
        else:
            with open(self.anno_file_path, 'r') as f:
                anno_data = json.load(f)
        self.pathes, self.labels, self.shapes, self.keys = [anno_data[x] for x in ['path', 'label', 'shape', 'key']]

    def complete_lmdb(self, out_dir):
        if os.path.isdir(out_dir):
            shard_num_list = []
            for file in sorted(glob(os.path.join(out_dir, 'shard*.lmdb'))):
                env = lmdb.open(file, subdir=False, lock=False, readonly=True)
                with env.begin(write=False) as txn:
                    shard_num_list.append(txn.stat()['entries'])
                env.close()                    
            complete_database = (sum(shard_num_list) == self.data_len) 
            return complete_database
        return False
        

    def init_db(self):
        self.db_list = []
        for file in self.db_files:
            db = lmdb.open(file,
                subdir=False,
                max_readers=self.db_cfg.max_readers,
                readonly=True,
                lock=False,
                readahead=self.db_cfg.readahead,
                meminit=self.db_cfg.meminit,
                max_spare_txns=self.db_cfg.max_spare_txns)
            self.db_list.append(db)

    def __getstate__(self):
        """Overwritten to allow pickling."""
        state = self.__dict__
        state["db"] = None
        return state

    def __setstate__(self, state):
        """Overwritten to allow pickling. The db handle will be regenerated after pickling."""
        self.__dict__ = state
        # Regenerate db handle after pickling:
        self.init_db()
    
    def __len__(self):
        return self.data_len

    def get_video_data(self, index):
        if self.db_list is None:
            self.init_db()
            assert(len(self.db_list))
        db_idx = index // self.num_per_shard
        if db_idx >= len(self.db_list):
            print(db_idx, len(self.db_list))
        with self.db_list[db_idx].begin(write=False) as txn:
            key = str(self.keys[index]).encode('ascii')
            byteflow = txn.get(key)

        # Tested this and the LMDB cannot be corrupted this way, even though byteflow is technically non-writeable
        # data_block = np.frombuffer(byteflow, dtype=np.uint8).reshape(self.shapes[index])
        # video = torch.from_numpy(data_block)
        video = torch.frombuffer(byteflow, dtype=torch.uint8).view(self.shapes[index])  #  C,T,H,W
        video = self.tfm_video(video, self.lmdb_crop, self.scale_resize, self.lmdb_aug, self.rand_flip, 
                        self.frame_num, self.split=='train')
        return video

    def __getitem__(self, index):
        video = self.get_video_data(index)
        return {'video':video, 'path': self.pathes[index], 'label': self.labels[index]}

    def _create_dataset(self, dataset, out_dir, db_cfg=LMDB_config,
            name_fmt='shard_{:03d}.lmdb', num_per_shard=int(1e4)):
        """Create the LMDB database from the given pytorch dataset.

        Each database entry is a blob of uint8 data describing each image.
        """
        os.makedirs(out_dir, exist_ok=True)
        num_shards_total = ceil(len(dataset) / num_per_shard)

        if platform.system() != "Linux" and db_cfg.map_size == 1099511627776 * 2:
            raise ValueError("Provide a reasonable default map_size for your operating system.")
        else:
            map_size = db_cfg.map_size

        cacheloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,   
            shuffle=db_cfg.shuffle_while_writing,
            drop_last=False,
            num_workers= torch.get_num_threads(),
            pin_memory=False,
            persistent_workers=False,
        )

        def open_lmdb(lmdb_path):
            db = lmdb.open(
                lmdb_path,
                subdir=False,
                map_size=map_size,
                readonly=False,
                meminit=db_cfg.meminit,
                writemap=True,
                map_async=True,
            )
            txn = db.begin(write=True)
            return db, txn

        # finalize dataset
        def close_lmdb(db, txn):
            if db and txn:
                txn.commit()
                db.close()

        idx, db, txn = 0, None, None
        pathes, label_list, shape_list = [], [], []
        for data in tqdm(cacheloader):
            for video, path, label in zip(data['video'], data['path'], data['label']):
                if idx % num_per_shard == 0:
                    close_lmdb(db, txn)
                    shard_num = 1 + idx // num_per_shard
                    lmdb_path = os.path.join(out_dir, name_fmt.format(shard_num))
                    log.info(f'Creating {lmdb_path} [{shard_num}/{num_shards_total}]...')
                    db, txn = open_lmdb(lmdb_path)
                # from torchvision.utils import save_image
                pathes.append(path)
                label_list.append(label.item())
                # video = np.asarray(video*255,dtype=np.uint8)
                video = self.video_tfm(video, path)
                # print(f'video_shape: {video.shape}')
                shape_list.append(video.shape)
                txn.put(str(idx).encode("ascii"), video.tobytes())
                idx += 1
        close_lmdb(db, txn)
        anno_dict = {'path': pathes, 'label': label_list, 
            'shape': shape_list, 'key': list(range(idx))}
        if self.anno_file_path.endswith('csv'):
            anno_df = pd.DataFrame.from_dict(anno_dict)
            anno_df.to_csv(self.anno_file_path)
        else:
            with open(self.anno_file_path, 'w') as f:
                json.dump(anno_dict, f)


    def video_tfm(self, video, path):
        return np.asarray(video*255,dtype=np.uint8)
    

class ExtraLmdbDataset(LmdbDataset):
    def __init__(self, dataset, args, can_create=True, db_cfg=LMDB_config):
        dump_vid_dir, codec = [args[x] for x in ['dump_vid_dir', 'codec', ]]
        dump_vid_dir, self.codec = os.path.expanduser(dump_vid_dir), codec.lower()
        assert self.codec in ['nerv', 'libx264', 'av1']
        if self.codec == 'nerv':
            # self.dump_vid_dir = os.path.join(dump_vid_dir, 'nerv')
            self.hypernerv_path = args['hypernerv']
            self.model = make_model(torch.load(self.hypernerv_path)['model'], load_sd=True)
            self.device = 'cuda:0'
            self.model.to(self.device)
            self.model.eval()
            self.quant_bits = int(args['crf'])
            self.nerv_decoder = None
            extra_str = f'{self.codec}_b{self.quant_bits}'
        else:
            crf, self.fps = args['crf'], args['fps']
            extra_str = f'{self.codec}_crf{crf}'
            self.option_dict = {'tune': 'fastdecode'} if self.codec=='libx264' else {'svtav1-params': 'fast-decode=3', 'cpu-used': str(8), 'usage': 'realtime', 'row-mt': '1'}
            self.option_dict.update({'crf': str(crf)})
        self.dump_vid_dir = os.path.join(dump_vid_dir, extra_str)
        os.makedirs(self.dump_vid_dir, exist_ok=True)
        args['name'] += f'_{extra_str}'
        self.orig_psnr_list, self.quant_psnr_list, self.new_psnr_list, self.bits_list = [], [], [], []
        super().__init__(dataset, args, can_create, db_cfg)
        with open(self.anno_file_path, 'r') as f:
            anno_data = json.load(f)            
        if len(self.orig_psnr_list):
            # update anno file, load and add psnr
            self.psnr_dict = self.avg_compress_psnr()
            self.avg_vid_size = {'avg_size_KB': sum(self.bits_list) / len(self.bits_list) / 8 /1e3}
            anno_dict = {**anno_data, **self.avg_vid_size, **self.psnr_dict}
            # import pdb; pdb.set_trace; from IPython import embed; embed()  
            with open(self.anno_file_path, 'w') as f:
                json.dump(anno_dict, f)
        else:
            self.psnr_dict = {k:anno_data[k] for k in ['avg_orig_psnr', 'avg_quant_psnr', 'avg_new_psnr']}
            self.avg_vid_size = {'avg_size_KB': anno_data['avg_size_KB']}

    @torch.no_grad()
    def video_tfm(self, video, path):
        if self.codec == 'nerv':
            # quantize nerv weights
            out_path = os.path.join(self.dump_vid_dir, os.path.split(path)[-1].replace('.mp4', '.pth'))
            out_dict=self.model(video[None].to(self.device))
            hyponet, w_modulate = out_dict['hyponet'], out_dict['w_modulate']
            quant_modulate_w, new_w = {}, {}
            for layer, mod_w in w_modulate.items():
                token_num, token_dim_list = hyponet.param_shapes[layer][-2:]
                modulate_w_shape = (1, token_num, sum(token_dim_list))
                quant_dict = quant_tensor(mod_w[0].view(modulate_w_shape), self.quant_bits)
                quant_modulate_w[layer], new_w[layer] = quant_dict['quant_t'], quant_dict['best_new_t']
            
            # dump nerv weights
            if not os.path.isfile(out_path):
                torch.save(quant_modulate_w, out_path)
            # compute actual size for nerv with quantization and entropy encoding
            self.bits_list.append(self.compute_total_bits(quant_modulate_w)['total_bits'])

            # dump modulate trace model
            batchsize_list = [2**x for x in range(10)]
            for device in ['cpu', self.device]:
                device_str = 'gpu' if device != 'cpu' else 'cpu'
                for batchsize in batchsize_list:
                    trace_ckt = os.path.join(self.dump_vid_dir, f'modulate_nerv_batch{batchsize}_{device_str}.pth')
                    if not os.path.exists(trace_ckt):
                        model = make_model(torch.load(self.hypernerv_path)['model'], load_sd=True)
                        trace_module = ModulateNeRV(model, video.size(1)).to(device)
                        trace_input = {k:v.repeat(batchsize,1,1).to(device) for k,v in new_w.items()}
                        torch.jit.save(torch.jit.trace(trace_module, trace_input), trace_ckt)
                        print(f'Dumped modulate nerv model to {trace_ckt}')

            # load decode nerv weights
            if self.nerv_decoder == None:
                modulate_nerv = os.path.join(self.dump_vid_dir, f'modulate_nerv_batch1_gpu.pth')
                self.nerv_decoder = torch.jit.load(modulate_nerv, map_location=self.device)

            new_video = self.nerv_decoder(new_w).cpu().detach()        # t,b,c,h,w
            new_video = new_video.squeeze(1).transpose(1,0)             # c,t,h,w
            # compute psnr 
            orig_psnr = self.compute_psnr(out_dict['pred_frames'].squeeze(1).transpose(1,0), video)
            quant_psnr = self.compute_psnr(new_video, video)
            new_video = np.asarray(new_video * 255., dtype=np.uint8)       # t,c,h,w -> c,t,h,w
            new_psnr = self.compute_psnr(torch.from_numpy(new_video).float()/255., video)
            self.orig_psnr_list.append(orig_psnr)
            self.quant_psnr_list.append(quant_psnr)
            self.new_psnr_list.append(new_psnr)
        else:
            out_path = os.path.join(self.dump_vid_dir, os.path.split(path)[-1])
            if not os.path.exists(out_path) or not cv2.VideoCapture(out_path).isOpened():
                new_video = (video*255).to(torch.uint8).permute(1,2,3,0)
                write_video(out_path, new_video, video_codec=self.codec, options=self.option_dict, fps=self.fps)        # T, H, W, C
            new_video = np.asarray(read_video(out_path, pts_unit='sec')[0].permute(-1,0,1,2))    # t,h,w,c -> c,t,h,w
            new_psnr = self.compute_psnr(torch.from_numpy(new_video).float()/255., video)
            self.new_psnr_list.append(new_psnr)
        if len(self.new_psnr_list) % 100 ==0:
            avg_psnr_dict = self.avg_compress_psnr()
            print(avg_psnr_dict) 

        return new_video
    
    def compute_psnr(self, pred, gt, max_value=1):
        if torch.cuda.is_available():
            pred, gt = pred.cuda(), gt.cuda()
        mse = F.mse_loss(pred.detach(), gt.detach(), reduction='none').mean(dim=(0,2,3))
        psnr = 10 * torch.log10((max_value ** 2) / mse).mean()
        return psnr.cpu().item()
    
    def avg_compress_psnr(self):
        psnr_lists = [self.orig_psnr_list, self.quant_psnr_list, self.new_psnr_list]
        name_list = ['avg_orig_psnr', 'avg_quant_psnr', 'avg_new_psnr']
        if self.codec != 'nerv':
            psnr_lists, name_list = psnr_lists[-1:], name_list[-1:]
        psnr_list = [sum(cur_psnr_list)/len(cur_psnr_list) for cur_psnr_list in psnr_lists]
        return {k:v for k,v in zip(name_list, psnr_list)}


    def compute_total_bits(self, quant_ckt):
        quant_v_list, tmin_scale_len = [], 0
        for k, layer_wt in quant_ckt.items():
            quant_v_list.extend(layer_wt['quant'].flatten().tolist())
            tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

        # get the element name and its frequency
        unique, counts = np.unique(quant_v_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(quant_v_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]

        # total bits for quantized embed + model weights
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        bits_per_param = total_bits / len(quant_v_list)
        
        # including the overhead for min and scale storage, 
        total_bits += tmin_scale_len * 16               #(16bits for float16)
        full_bits_per_param = total_bits / len(quant_v_list)

        out_dict = {'total_bits': total_bits, 'bits_per_param': bits_per_param, 'full_bits_per_param': full_bits_per_param}
        return out_dict
