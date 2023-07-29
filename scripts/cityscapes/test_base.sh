export OMP_NUM_THREADS=1

python test.py --model_type=base_128x256 \
    --checkpoint_path /path/to/checkpoint \
    --end_sample_idx=4 \
    --test_batch_size=4 \
    --dataset cityscapes \
    --save_path=results/sample.png \
    --test_captions "An image of an urban street view with Buildings, Fences, Roads, People, Skies, Traffic signs, Poles, Vegetations, Sidewalks and Cars." \
    ${@:1}