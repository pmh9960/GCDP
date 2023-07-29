export OMP_NUM_THREADS=1

accelerate launch --config_file scripts/celeba/train_sr_128x128_256x256.yaml \
    train.py \
    --project imagen-celeba \
    --root_dir /path/to/data \
    --caption_list_dir data/caption_files/celeba \
    --test_caption_files data/eval_samples/celeba/celeba_val_1000_caption.txt \
    --dataset celeba \
    --num_classes 19 \
    --exp_name sr_128x128_256x256 \
    --model_type sr_128x128_256x256 \
    --random_crop_size 256 256 \
    --num_iters 300000 \
    --log_every 10000 \
    --save_every 10000 \
    --max_batch_size 4 \
    --batch_size 4 \
    --checkpoint_dir checkpoints \
    --test_batch_size 1 \
    --augmentation_type flip \
    --start_image_or_video data/eval_samples/celeba/celeba_val_1000_img.jpg \
    --start_label_or_video data/eval_samples/celeba/celeba_val_1000_lbl.png \
    --fp16 ${@:1}