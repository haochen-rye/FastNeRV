import os
import shutil
import subprocess
from math import ceil

###############################################################################################
code_dir = '/data/home/haochen1/nerv_projects/GenNeRV'
script_dir = f'{code_dir}/scripts/parallel_scripts/batch'

std_cmd = "cd {code_dir}\n\
python run_trainer.py --cfg cfgs/tnerv_enc.yaml \
   --csv_file {csv_file} --out_path outputs/{out_dir} {wandb} \
   --frame_num {frames} --t_duration {t_duration} --input_size {input_size} \
   -b {batch_size} --tag {tag} -p {port} \
   -j {num_workers} --opts train_dataset.args.cls_vid_num {cls_vid_num}  \
   zero_center {zero_center}  \
   train_dataset.args.resize {resize}  \
   train_dataset.args.rand_flip {rand_flip}  \
   model.args.tokenizer.args.patch_t {patch_t}   \
   model.args.tokenizer.args.patch_hw {patch_hw}   \
   model.args.tokenizer.args.learn_pe {enc_learn_pe}   \
    model.args.encoder.name ViT-{enc_vit} \
    model.args.decoder.name ViT-{dec_vit} \
    model.args.decoder.expand_decoder {expand_decoder} \
    model.args.decoder.learn_p_pe {learn_p_pe} \
    model.args.decoder.learn_z_pe {learn_z_pe} \
    model.args.decoder.token_init_std {token_init_std} \
    model.args.decoder.z_bottleneck {z_bottleneck} \
    model.args.decoder.z_tokens {z_tokens} \
    model.args.decoder.z_dim {z_dim} \
    model.args.decoder.z_norm {z_norm} \
    model.args.decoder.z_norm_first {z_norm_first} \
    model.args.decoder.use_v_tokens {use_v_tokens} \
    model.args.decoder.v_dim {v_dim} \
    model.args.decoder.sequential_decode {sequential_decode} \
    model.args.decoder.cumsum_z {cumsum_z} \
    model.args.decoder.d_tokens {d_tokens} \
    model.args.decoder.final_ch {final_ch} \
    model.args.decoder.final_ks {final_ks} \
    model.args.decoder.out_bias {out_bias} \
    model.args.decoder.out_range {out_range} \
    model.args.kl_layer.deterministic {deterministic} \
    model.args.kl_layer.kl_weight {kl_weight} \
    model.args.kl_layer.avg_kl {avg_kl} \
    model.args.vq_layer.use_vq {use_vq} \
    model.args.vq_layer.codebook_size {codebook_size} \
    model.args.vq_layer.vq_weight {vq_weight} \
    model.args.vq_layer.commit_loss {commit_loss} \
    model.args.vq_layer.codebook_loss {codebook_loss} \
    model.args.vq_layer.ema {ema} \
    model.args.vq_layer.ema_type {ema_type} \
   optimizer.name {optimizer} \
   optimizer.args.lr {lr} \
   optimizer.lr_type {lr_type} \
   optimizer.epoch_cycles {epoch_cycles} \
   optimizer.cycle_decay {cycle_decay} \
   optimizer.full_cosine {full_cosine} \
   optimizer.loss_weights {loss_weights} \
   log_every_n {log_every_n} \
   vis_epoch {vis_epoch} vis_n {vis_num} resume_ckt {resume_ckt} dump_pred {dump_img} \
   eval_epoch {eval_epoch}  max_epoch {max_epoch} save_epoch {save_epoch} "

prefix, gpu_num, cpu_num, wandb = 'ha3', 8, 0, '-w'
out_dir, cls_vid_num, input_size, frames, fps = f'421_{prefix}', '400_125', 128, 16, 16
batch_size, num_workers, zero_center, out_bias, out_range, resize, rand_flip = 32, 8*gpu_num, True, 0, 'no', False, True
csv_file = ['ucf_101.csv', 'k400_short192.csv', 'k400_short128.csv', 'ucf_101_short128.csv'][-2]
patch_t, patch_hw, z_tokens, z_dim, d_tokens, z_bottleneck, token_init_std = 1, 32, 4, 48, 0, True, 1
expand_decoder, use_v_tokens, v_dim, sequential_decode, cumsum_z = True, '4_4_4', 6, False, False
enc_vit, dec_vit, learn_p_pe, learn_z_pe, final_ch, final_ks = 'B', 'B', True, False, 4, 3
enc_learn_pe, deterministic, kl_weight, avg_kl, z_norm, z_norm_first = True, True, 5e-7, False, 'bn_bn', True
use_vq, codebook_size, vq_weight, commit_loss, codebook_loss, ema, ema_type = False, 8192, 1e-1, 0.25, 1, False, 'v'
lr, optimizer, loss_weights, lr_type, full_cosine, cycle_decay =  1e-4, 'adamw', '1_0_0', 'step', False, 1
base_port, resume_ckt, vis_num, max_epoch, epoch_cycles, log_every_n = 5100, 'no', 16, 100, 1, 100
if 'ucf' in csv_file:
    eval_epoch, vis_epoch, save_epoch = 30, 30, 100
else:
    eval_epoch, vis_epoch, save_epoch = 15, 15, 50
partition_list = ['ai'] * 3 + ['ai4p'] * 12

setup_list = []    
patch_t, patch_hw, use_v_tokens, v_dim, z_tokens, z_dim, use_vq, deterministic = \
    2, 32, '8_4_4', 12, 16, 96, False, True
for patch_t, patch_hw, use_v_tokens, z_tokens, z_dim in [(2, 32, '8_4_4', 4, 384), (2, 32, '8_4_4', 16, 96),
     (2, 32, '8_4_4', 32, 48), (1, 32, '8_4_4', 8), (2, 16, '8_4_4', 8),][:3]:
    setup_list.append((patch_t, patch_hw, use_v_tokens, v_dim, z_tokens, z_dim, use_vq, deterministic))

# for z_tokens, z_dim in [(0, 48), (2, 384), (4, 192), (8, 192), (4, 384)]:

# for use_vq, deterministic in [(False, True), (True, True), (False, False)]:

# z_tokens, z_dim,  patch_t, patch_hw, use_v_tokens, v_dim, codebook_size, commit_loss, vq_weight, deterministic, kl_weight = \
#     4, 48, 4, 32, '4_4_4', 6, 4096, 0.25, 1e-1, True, 1e-7

# for z_tokens, z_dim in [(4, 48), (2, 96), (8, 24), (0, 48)]:

# for patch_t, patch_hw, use_v_tokens, v_dim in [(4, 32, '4_4_4', 6), (4, 32, '4_4_4', 4), (4, 32, '4_4_4', 8), 
#     (2, 32, '8_4_4', 6), (8, 32, '2_4_4', 6), (4, 16, '4_8_8', 6), (4, 64, '4_2_2', 6),
#     (2, 32, '8_4_4', 3), (8, 32, '2_4_4', 12), (4, 16, '4_8_8', 2), (4, 64, '4_2_2', 24), (4, 64, '4_2_2', 12)]:

# for codebook_size in [2048, 4096, 9012]:

# for commit_loss in [0.1, 0.25, 1]:

# for vq_weight in [1, 1e-1, 1e-2]:

# for deterministic, kl_weight in [(False, 1e-8), (False, 1e-7), (False, 1e-6)]:


cmd_list, idx = [], 0
for patch_t, patch_hw, use_v_tokens, v_dim, z_tokens, z_dim, use_vq, deterministic in set(setup_list):
    tag=f'{out_dir}_{idx+1:02d}'
    out_bias, out_range = (0, 'neg') if zero_center else (0.5, 'zero') 
    cur_cmd = std_cmd.format(code_dir=code_dir, out_dir=out_dir, tag=tag, port=idx+base_port,
        frames=frames, batch_size=batch_size, num_workers=num_workers, wandb=wandb, 
        csv_file=csv_file, lr_type=lr_type, t_duration=frames/fps,  z_dim=z_dim, z_tokens=z_tokens,
        use_v_tokens=use_v_tokens, v_dim=v_dim, sequential_decode=sequential_decode, cumsum_z=cumsum_z,
        use_vq=use_vq, codebook_size=codebook_size, vq_weight=vq_weight, commit_loss=commit_loss,
        codebook_loss=codebook_loss, ema=ema, ema_type=ema_type, log_every_n=log_every_n,
        expand_decoder=expand_decoder, final_ch=final_ch, final_ks=final_ks, d_tokens=d_tokens,
        input_size=input_size, cls_vid_num=cls_vid_num, out_bias=out_bias, enc_vit=enc_vit, dec_vit=dec_vit, 
        patch_t=patch_t, patch_hw=patch_hw, z_bottleneck=z_bottleneck, token_init_std=token_init_std,
        enc_learn_pe=enc_learn_pe, learn_p_pe=learn_p_pe, learn_z_pe=learn_z_pe,
        lr=lr, deterministic=deterministic, optimizer=optimizer, loss_weights=loss_weights,
        kl_weight=kl_weight, vis_epoch=vis_epoch, vis_num=vis_num, eval_epoch=eval_epoch, max_epoch=max_epoch, 
        save_epoch=save_epoch, resume_ckt=resume_ckt, avg_kl=avg_kl, z_norm=z_norm, z_norm_first=z_norm_first,
        epoch_cycles=epoch_cycles, cycle_decay=cycle_decay, full_cosine=full_cosine,
        dump_img='no', resize=resize, rand_flip=rand_flip, zero_center=zero_center, out_range=out_range,
        )
    cmd_list.append(cur_cmd)
    idx += 1

os.makedirs(script_dir, exist_ok=True)
os.chdir(script_dir)
if os.path.isdir(prefix):
    shutil.rmtree(prefix)
os.makedirs(prefix, exist_ok=True)

start_idx, cmd_num = 0, 1
script_num = ceil(len(cmd_list) / cmd_num )
print(f'Total cmd_num: {len(cmd_list)}, script_num: {script_num}')
submit_cmd_list = []
for i in range(script_num):
    end_idx = start_idx + cmd_num
    temp_file = os.path.join(prefix, f'{i+1:02d}')
    cur_partition = partition_list[i]
    if i+1 >= 0:
        print(f'Creating {temp_file}...')
        cur_cmd_list = cmd_list[start_idx:end_idx]
        with open(temp_file, 'w') as f:
            f.write('\n'.join(cur_cmd_list))
        submit_command = f'submit {temp_file} --nog {gpu_num} --no_output -p {cur_partition} --cpu {cpu_num}'
        submit_cmd_list.append(submit_command)
    start_idx = end_idx

for submit_cmd in submit_cmd_list:
    print(f'execute command: {submit_cmd}')
    subprocess.run(submit_cmd.split())            
    