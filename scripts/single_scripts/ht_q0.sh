# x4k1kfps
cd /data/home/haochen1/nerv_projects/GenNeRV
python run_trainer.py --replace --cfg cfgs/tnerv_enc.yaml  \
    --csv_file k400_short128.csv --out_path outputs/dbg/421_h1 \
    --frame_num 15 --t_duration 1.0 --input_size 128 \
    -b 32 --tag 01 -j 8 -p 48 \
    --opts train_dataset.args.cls_vid_num 400_1e6   \
    model.args.kl_layer.deterministic True \
    model.args.kl_layer.kl_weight 1e-8 \
    model.args.vq_layer.use_vq True \
    model.args.vq_layer.codebook_size 4096 \
    model.args.vq_layer.vq_weight 1e-1 \
    model.args.vq_layer.codebook_loss 1. \
    model.args.vq_layer.commit_loss 0.25 \
    model.args.vq_layer.unuse_token_thres 0. \
    model.args.vq_layer.ema False \
    model.args.vq_layer.using_znorm False \
    model.args.encoder.name ViT-B \
    model.args.decoder.name ViT-B \
    model.args.decoder.z_bottleneck True \
    model.args.decoder.z_tokens 4 \
    model.args.tokenizer.args.patch_t 3 \
    model.args.tokenizer.args.patch_hw 32 \
    model.args.decoder.z_dim 32 \
    model.args.decoder.use_v_tokens 5_4_4 \
    model.args.decoder.v_dim 4 \
    model.args.decoder.z_norm bn_bn \
    model.args.decoder.z_norm_first True \
    model.args.decoder.imgwise_decode False \
    model.args.decoder.sequential_decode False \
    model.args.decoder.cumsum_z False \
    model.args.decoder.final_ch 4 \
    model.args.decoder.final_ks 3 \
    model.args.decoder.out_bias 0.5 \
    zero_center False \
    optimizer.name adamw    optimizer.args.lr 1e-4  \
    optimizer.lr_type step \
    log_every_n 25 \
    vis_epoch 1 vis_n 4  eval_epoch 1  max_epoch 3 save_epoch 1 \
    dump_pred no dump_video no dump_vid_grid yes

# 1_0_0_4_0_0_16_0_0