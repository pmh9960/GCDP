export OMP_NUM_THREADS=1

accelerate launch --config_file scripts/cityscapes/train_sr_128x256_256x512.yaml \
    train.py \
    --root_dir /path/to/data \
    --caption_list_dir data/caption_files/cityscapes \
    --test_caption_files data/eval_samples/cityscapes/frankfurt_000000_000294.txt \
    --exp_name sr_128x256_256x512 \
    --model_type sr_128x256_256x512 \
    --num_iters 300000 \
    --log_every 10000 \
    --save_every 10000 \
    --max_batch_size 8 \
    --batch_size 8 \
    --random_crop_size 256 512 \
    --checkpoint_dir checkpoints \
    --test_batch_size 1 \
    --start_image_or_video data/eval_samples/cityscapes/frankfurt_000000_000294_leftImg8bit.png \
    --start_label_or_video data/eval_samples/cityscapes/frankfurt_000000_000294_gtFine_labelIds.png \
    --augmentation_type=resizedCrop_1.0_1.2 \
    --fp16 ${@:1}