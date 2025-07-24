import os
import shutil
import subprocess
from math import ceil

###############################################################################################
code_dir = '/data/home/haochen1/nerv_projects/GenNeRV'
script_dir = f'{code_dir}/scripts/parallel_scripts/batch'

std_cmd = "cd {code_dir}\n\
python run_trainer.py --cfg cfgs/param_enc.yaml \
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
    model.args.decoder.learn_p_pe {learn_p_pe} \
    model.args.decoder.learn_z_pe {learn_z_pe} \
    model.args.decoder.token_init_std {token_init_std} \
    model.args.decoder.z_bottleneck {z_bottleneck} \
    model.args.decoder.z_tokens {z_tokens} \
    model.args.decoder.z_dim {z_dim} \
    model.args.decoder.z_norm {z_norm} \
    model.args.decoder.z_norm_first {z_norm_first} \
    model.args.decoder.use_v_tokens {use_v_tokens} \
    model.args.decoder.d_tokens {d_tokens} \
    model.args.decoder.final_ch {final_ch} \
    model.args.decoder.final_ks {final_ks} \
    model.args.decoder.out_bias {out_bias} \
    model.args.decoder.out_range {out_range} \
    model.args.kl_layer.deterministic {deterministic} \
    model.args.kl_layer.kl_weight {kl_weight} \
    model.args.kl_layer.avg_kl {avg_kl} \
   optimizer.name {optimizer} \
   optimizer.args.lr {lr} \
   optimizer.lr_type {lr_type} \
   optimizer.epoch_cycles {epoch_cycles} \
   optimizer.cycle_decay {cycle_decay} \
   optimizer.full_cosine {full_cosine} \
   optimizer.loss_weights {loss_weights} \
   vis_epoch {vis_epoch} vis_n {vis_num} resume_ckt {resume_ckt} dump_pred {dump_img} \
   eval_epoch {eval_epoch}  max_epoch {max_epoch} save_epoch {save_epoch} "

prefix, gpu_num, cpu_num, wandb = 'pa3', 4, 0, '-w'
out_dir, cls_vid_num, input_size, frames, fps = f'401_{prefix}', '400_1e4', 128, 16, 16
batch_size, num_workers, zero_center, out_bias, out_range, resize, rand_flip = 16, 8*gpu_num, True, 0, 'neg', False, True
csv_file = ['ucf_101.csv', 'k400_short192.csv', 'k400_short128.csv', 'ucf_101_short128.csv'][-2]
patch_t, patch_hw, z_tokens, z_dim, d_tokens, z_bottleneck, token_init_std = 1, 32, 64, 256, 0, True, 1
enc_vit, dec_vit, learn_p_pe, learn_z_pe, use_v_tokens, final_ch, final_ks = 'B', 'S', True, False, False, 4, 3
enc_learn_pe, deterministic, kl_weight, avg_kl, z_norm, z_norm_first = True, False, 5e-7, False, 'bn_bn', True
lr, optimizer, loss_weights, lr_type, full_cosine, cycle_decay =  1e-4, 'adamw', '1_0_0', 'step', False, 1
base_port, resume_ckt, vis_num, max_epoch, epoch_cycles = 5100, 'no', 16, 150, 1
if 'ucf' in csv_file:
    eval_epoch, vis_epoch, save_epoch = 30, 30, 100
else:
    eval_epoch, vis_epoch, save_epoch = 15, 15, 15
partition_list = ['ai'] * 20 + ['ai4p'] * 12

setup_list = []    
z_tokens, z_dim, kl_weight = 64, 256, 5e-7
for kl_weight in [1e-7, 2e-8, 5e-7]:
    for z_tokens, z_dim in [(64, 256), (32, 256), ]:
        setup_list.append((z_tokens, z_dim, kl_weight))

cmd_list, idx = [], 0
for z_tokens, z_dim, kl_weight in set(setup_list):
    tag=f'{out_dir}_{idx+1:02d}'
    out_bias, out_range = (0, 'neg') if zero_center else (0.5, 'zero') 
    cur_cmd = std_cmd.format(code_dir=code_dir, out_dir=out_dir, tag=tag, port=idx+base_port,
        frames=frames, batch_size=batch_size, num_workers=num_workers, wandb=wandb, 
        csv_file=csv_file, lr_type=lr_type, t_duration=frames/fps,  z_dim=z_dim, z_tokens=z_tokens,
        use_v_tokens=use_v_tokens, final_ch=final_ch, final_ks=final_ks, d_tokens=d_tokens,
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
    