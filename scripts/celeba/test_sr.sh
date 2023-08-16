export OMP_NUM_THREADS=1

python test.py --model_type=sr_128x128_256x256 \
    --start_at_unet_number 2 \
    --cond_scale 1.0 \
    --noise_schedules linear linear \
    --noise_schedules_lbl linear linear \
    --checkpoint_path /path/to/checkpoint \
    --end_sample_idx=1 \
    --test_batch_size=1 \
    --dataset celeba \
    --num_classes 19 \
    --save_path=results/celeba/sr.png \
    --test_captions "The person has mouth slightly open, high cheekbones, and double chin and wears hat, and earrings." \
    --start_image_or_video=data/eval_samples/celeba/celeba_val_1000_img.jpg \
    --start_label_or_video=data/eval_samples/celeba/celeba_val_1000_lbl.png \
    ${@:1}