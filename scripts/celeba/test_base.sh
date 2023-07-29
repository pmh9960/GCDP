export OMP_NUM_THREADS=1

python test.py --model_type=base_128x128 \
    --checkpoint_path /path/to/checkpoint \
    --end_sample_idx=4 \
    --test_batch_size=4 \
    --dataset celeba \
    --num_classes 19 \
    --save_path=results/sample.png \
    --test_captions "The woman wears earrings. She has wavy hair. She is attractive." \
    ${@:1}