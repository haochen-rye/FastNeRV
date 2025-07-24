# x4k1kfps
cd /data/home/haochen1/nerv_projects/GenNeRV
python run_trainer.py --replace --cfg cfgs/htnerv_enc.yaml  \
    --csv_file k400_short128.csv --out_path outputs/dbg/410_h11 \
    --frame_num 15 --t_duration 1.0 --input_size 128 \
    -b 32 --tag 01 -j 8 -p 45 \
    --opts train_dataset.args.cls_vid_num 400_1e6   \
    model.args.tokenizer.args.patch_t 1 \
    model.args.tokenizer.args.patch_hw 32 \
    model.args.kl_layer.deterministic True \
    model.args.kl_layer.kl_weight 1e-8 \
    model.args.encoder.name ViT-B \
    model.args.decoder.name ViT-B \
    model.args.decoder.z_tokens 1_0_0_4_0_0_16_0_0_64 \
    model.args.decoder.z_dim 256 \
    model.args.decoder.use_v_tokens True \
    model.args.decoder.sequential_decode True \
    model.args.decoder.cumsum_z False \
    model.args.decoder.final_ch 4 \
    model.args.decoder.final_ks 3 \
    model.args.decoder.out_bias 0.5 \
    optimizer.name adamw    optimizer.args.lr 0.0001  \
    optimizer.lr_type step \
    vis_epoch 1 vis_n 4  eval_epoch 1  max_epoch 3 save_epoch 1 \
    dump_pred no dump_video no dump_vid_grid yes
