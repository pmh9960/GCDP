export OMP_NUM_THREADS=1

accelerate launch --config_file scripts/cityscapes/train_joint_base_128x256.yaml \
    train.py \
    --root_dir /path/to/data \
    --caption_list_dir data/caption_files/cityscapes \
    --test_caption_files data/eval_samples/cityscapes/frankfurt_000000_000294.txt \
    --exp_name base_128x256 \
    --model_type base_128x256 \
    --num_iters 300000 \
    --log_every 10000 \
    --save_every 10000 \
    --max_batch_size 8 \
    --batch_size 8 \
    --checkpoint_dir checkpoints \
    --test_batch_size 4 \
    --augmentation_type=resizedCrop_1.0_1.2 \
    --noise_schedules linear linear --noise_schedules_lbl linear linear \
    --fp16 ${@:1}