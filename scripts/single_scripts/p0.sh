# x4k1kfps
cd /data/home/haochen1/nerv_projects/GenNeRV
python run_trainer.py --replace --cfg cfgs/param_enc.yaml  \
    --csv_file k400_short128.csv --out_path outputs/dbg/331_i0 \
    --frame_num 15 --t_duration 1.0 --input_size 128 \
    -b 32 --tag 01 -j 8 -p 44 \
    --opts train_dataset.args.cls_vid_num 20_20   \
    model.args.tokenizer.args.patch_t 3 \
    model.args.kl_layer.deterministic False \
    model.args.kl_layer.kl_weight 1e-8 \
    model.args.encoder.name ViT-B \
    model.args.decoder.name ViT-S \
    model.args.decoder.z_tokens 128 \
    model.args.decoder.z_dim 256 \
    model.args.decoder.use_v_tokens 1_4_4 \
    model.args.decoder.final_ch 4 \
    model.args.decoder.final_ks 3 \
    model.args.decoder.out_bias 0.5 \
    optimizer.name adamw    optimizer.args.lr 0.0001  \
    optimizer.lr_type step \
    vis_epoch 1 vis_n 4  eval_epoch 1  max_epoch 3 save_epoch 1 \
    dump_pred no dump_video no dump_vid_grid yes
