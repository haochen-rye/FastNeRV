import os
import shutil
import subprocess
from math import ceil

###############################################################################################
code_dir = '/data/home/haochen1/nerv_projects/GenNeRV'
script_dir = f'{code_dir}/scripts/parallel_scripts/batch'

std_cmd = "cd {code_dir}\n\
python run_trainer.py --cfg cfgs/nerv_enc.yaml \
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
   model.args.hyponet.args.out_bias {out_bias}  \
   model.args.hyponet.args.out_range {out_range}  \
   model.args.hyponet.args.strds {strds}   \
   model.args.hyponet.args.ks {ks}   \
   model.args.hyponet.args.act {act}   \
   model.args.hyponet.args.hid_dim {hid_dim}  \
   model.args.hyponet.args.pos_embed {nerv_pe} \
   model.args.hyponet.args.pe_dim {nerv_pe_dim}   \
   model.args.hyponet.args.learn_pe {nerv_learn_pe}   \
   model.args.nerv_tokens.token_nums {token_nums}  \
   model.args.nerv_tokens.token_dims {token_dims} \
   model.args.nerv_tokens.token_init_std {token_init_std} \
   model.args.nerv_tokens.z_type {z_type} \
   model.args.nerv_tokens.z_dim {z_dim} \
   model.args.nerv_tokens.z_norm {z_norm} \
   model.args.nerv_tokens.z_norm_first {z_norm_first} \
   model.args.nerv_tokens.repeat_dim {repeat_dim} \
    model.args.kl_layer.deterministic {deterministic} \
    model.args.kl_layer.kl_weight {kl_weight} \
    model.args.kl_layer.avg_kl {avg_kl} \
    model.args.transformer_encoder.name ViT-{dit} \
    model.args.transformer_encoder.proj_drop {proj_drop} \
    model.args.transformer_encoder.attn_drop {attn_drop} \
    model.args.transformer_encoder.mlp_drop {mlp_drop} \
   optimizer.name {optimizer} \
   optimizer.args.lr {lr} \
   optimizer.lr_type {lr_type} \
   optimizer.epoch_cycles {epoch_cycles} \
   optimizer.cycle_decay {cycle_decay} \
   optimizer.full_cosine {full_cosine} \
   optimizer.loss_weights {loss_weights} \
   vis_epoch {vis_epoch} vis_n {vis_num} resume_ckt {resume_ckt} dump_pred {dump_img} \
   eval_epoch {eval_epoch}  max_epoch {max_epoch} save_epoch {save_epoch} "


prefix, gpu_num, cpu_num, wandb = 'n0', 4, 0, '-w'
out_dir, cls_vid_num, input_size, frames, fps = f'329_{prefix}', '400_1e6', 128, 16, 16
batch_size, num_workers, zero_center, out_bias, out_range = 16, 8*gpu_num, True, 0, 'neg'
csv_file = ['ucf_101.csv', 'k400_short192.csv', 'k400_short128.csv', 'ucf_101_short128.csv'][-2]
patch_t, patch_hw, strds, ks, act = 1, 32, '4_8_2_2', '1_3', 'gelu'
hid_dim, resize, rand_flip = 64, False, True
token_nums, token_dims, z_type, z_dim = '0_64_0_0', '1024_1152_1152_1152', 'nerv', 256
dit, lr, optimizer, loss_weights, lr_type, full_cosine, cycle_decay = 'B', 1e-4, 'adamw', '1_0_0', 'step', False, 1
enc_learn_pe, nerv_pe, nerv_pe_dim, nerv_learn_pe = True,  'inr_1024', 16, True
token_init_std, z_norm, z_norm_first, repeat_dim = 1, 'no', False, -2
proj_drop, attn_drop, mlp_drop, base_port, resume_ckt = 0, 0, 0, 5200, 'no'
deterministic, kl_weight, avg_kl, vis_num, max_epoch, epoch_cycles  = False, 1e-7, False, 16, 150, 1
if 'ucf' in csv_file:
    eval_epoch, vis_epoch, save_epoch = 30, 30, 100
else:
    eval_epoch, vis_epoch, save_epoch = 15, 15, 30
partition_list = ['ai'] * 20 + ['ai4p'] * 12

setup_list = []    
# enc_learn_pe, zero_center, out_bias, out_range, nerv_pe, nerv_pe_dim, nerv_learn_pe, token_init_std, lr, patch_hw, lr_type, repeat_dim = \
#     True, True, 0, 'neg', 'fix_1024', 16, True, 0.02, 1.5e-4, 16, 'cosine', '-2'
nerv_learn_pe, z_norm, z_norm_first, token_nums, z_dim = 32, 1e-7, True, 'bn_bn', False, '0_64_0_0', 256
for patch_hw in 
for kl_weight in [1e-7, 1e-6]:
    setup_list.append((kl_weight, nerv_learn_pe, z_norm, z_norm_first, token_nums, z_dim))

cmd_list, idx = [], 0
for kl_weight, nerv_learn_pe, z_norm, z_norm_first, token_nums, z_dim in set(setup_list):
    tag=f'{out_dir}_{idx+1:02d}_PE_E{enc_learn_pe}_N{nerv_learn_pe}_Out{out_range}_init{token_init_std}_Repeat{repeat_dim}'
    cur_cmd = std_cmd.format(code_dir=code_dir, out_dir=out_dir, tag=tag, port=idx+base_port,
        frames=frames, batch_size=batch_size, num_workers=num_workers, wandb=wandb, 
        csv_file=csv_file, lr_type=lr_type, t_duration=frames/fps, z_type=z_type, z_dim=z_dim,
        input_size=input_size, cls_vid_num=cls_vid_num, out_range=out_range, out_bias=out_bias, 
        patch_t=patch_t, patch_hw=patch_hw, strds=strds, ks=ks, act=act, 
        hid_dim=hid_dim, token_nums=token_nums, token_dims=token_dims, 
        lr=lr, deterministic=deterministic, optimizer=optimizer, loss_weights=loss_weights,
        kl_weight=kl_weight, vis_epoch=vis_epoch, vis_num=vis_num, eval_epoch=eval_epoch, max_epoch=max_epoch, 
        save_epoch=save_epoch, resume_ckt=resume_ckt, dit=dit, avg_kl=avg_kl,
        epoch_cycles=epoch_cycles, cycle_decay=cycle_decay, full_cosine=full_cosine,
        dump_img='no', resize=resize, rand_flip=rand_flip, z_norm=z_norm, repeat_dim=repeat_dim, 
        z_norm_first=z_norm_first, nerv_pe=nerv_pe, nerv_pe_dim=nerv_pe_dim, nerv_learn_pe=nerv_learn_pe,
        token_init_std=token_init_std, zero_center=zero_center, enc_learn_pe=enc_learn_pe,
        proj_drop=proj_drop, attn_drop=attn_drop, mlp_drop=mlp_drop, 
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
    