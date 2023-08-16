export OMP_NUM_THREADS=1

python test.py --model_type=sr_128x256_256x512 \
    --start_at_unet_number 2 \
    --cond_scale 1.0 \
    --noise_schedules linear linear \
    --noise_schedules_lbl linear linear \
    --checkpoint_path /path/to/checkpoint \
    --end_sample_idx=1 \
    --test_batch_size=1 \
    --dataset cityscapes \
    --num_classes 20 \
    --save_path=results/cityscapes/sr.png \
    --test_captions "An image of an urban street view with Buildings, Fences, Roads, People, Skies, Traffic signs, Poles, Vegetations, Sidewalks and Cars." \
    --start_image_or_video=data/eval_samples/cityscapes/frankfurt_000000_000294_leftImg8bit.png \
    --start_label_or_video=data/eval_samples/cityscapes/frankfurt_000000_000294_gtFine_labelIds.png \
    ${@:1}