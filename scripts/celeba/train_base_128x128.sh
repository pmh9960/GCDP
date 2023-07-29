export OMP_NUM_THREADS=1

accelerate launch --config_file scripts/celeba/train_base_128x128.yaml \
    train.py \
    --root_dir /path/to/data \
    --caption_list_dir data/caption_files/celeba \
    --test_caption_files data/eval_samples/celeba/celeba_caption_57.txt \
    --dataset celeba \
    --num_classes 19 \
    --exp_name base_128x128 \
    --model_type base_128x128 \
    --num_iters 300000 \
    --log_every 10000 \
    --save_every 10000 \
    --max_batch_size 8 \
    --batch_size 8 \
    --checkpoint_dir checkpoints \
    --test_batch_size 4 \
    --augmentation_type flip \
    --fp16 ${@:1}
