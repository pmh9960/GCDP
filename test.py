import argparse
import os
import os.path as osp

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

from datasets.celeba import mapping_id as mapping_id_celeba
from datasets.celeba import transform_lbl as transform_lbl_celeba
from datasets.cityscapes import ToTensorNoNorm
from datasets.cityscapes import id_type_to_classes as id_type_to_classes_cityscapes
from datasets.cityscapes import transform_lbl as transform_lbl_cityscapes
from imagen_pytorch import BaseJointUnet, JointImagen, JointImagenTrainer, SRJointUnet
from imagen_pytorch.imagen_pytorch import NullUnet


def read_jsonl(jsonl_path):
    import jsonlines
    lines = []
    with jsonlines.open(jsonl_path, 'r') as f:
        for line in f.iter():
            lines.append(line)
    return lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, nargs='+', required=True)
    parser.add_argument('--save_path', type=str, required=True)

    parser.add_argument('--lowres_dir', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=20)
    # cityscapes: 20, celeba: 19 (include background)

    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--start_sample_idx', type=int, default=0, help='included')
    parser.add_argument('--end_sample_idx', type=int, default=2975, help='not included')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--test_captions', type=str, nargs='*', default=['', ])
    parser.add_argument('--caption_list_dir', type=str, default='')

    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--sample_timesteps', type=int, default=100)
    parser.add_argument('--cond_scale', type=float, nargs='+', default=(3.0, ))
    parser.add_argument('--lowres_sample_noise_level', type=float, default=0.2)
    parser.add_argument('--start_image_or_video', type=str,
                        default='samples/frankfurt_000000_000294_leftImg8bit.png')
    parser.add_argument('--start_label_or_video', type=str,
                        default='samples/frankfurt_000000_000294_gtFine_labelIds.png')
    parser.add_argument('--return_all_unet_outputs', action='store_true')
    parser.add_argument('--start_at_unet_number', type=int, default=1)
    parser.add_argument('--stop_at_unet_number', type=int, default=3)

    parser.add_argument('--noise_schedules', type=str, nargs='*', default=('cosine', ))
    parser.add_argument('--noise_schedules_lbl', type=str, nargs='*', default=('cosine_p', ))
    parser.add_argument('--cosine_p_lbl', type=float, default=1.0)

    parser.add_argument('--channels_lbl', type=int, default=3)
    parser.add_argument('--pred_objectives', type=str, default='noise')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    parser.add_argument('--condition_on_text', action='store_true')
    parser.add_argument('--no_condition_on_text', action='store_false', dest='condition_on_text')
    parser.set_defaults(condition_on_text=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no_fp16', action='store_false', dest='fp16')
    parser.set_defaults(fp16=True)

    args = parser.parse_args()

    args.cond_scale = args.cond_scale[0] if len(args.cond_scale) == 1 else args.cond_scale

    if len(args.test_captions) == 1 and args.test_batch_size != 1:
        args.test_captions = args.test_captions * args.test_batch_size
    assert len(args.test_captions) == args.test_batch_size, \
        (len(args.test_captions), args.test_batch_size)

    print(f'Sample Indices: {args.start_sample_idx} - {args.end_sample_idx}')

    return args


def main():
    args = parse_args()

    # unet for imagen

    print(f'Creating JointUNets.. {args.model_type}')

    start_at_unet_number = args.start_at_unet_number
    stop_at_unet_number = args.stop_at_unet_number
    if args.model_type.startswith('base'):
        addi_kwargs = dict()
        addi_kwargs.update(dict(
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True)
            if args.condition_on_text else False,
        ))
        unet1 = BaseJointUnet(channels_lbl=args.channels_lbl, num_classes=args.num_classes, **addi_kwargs)
        unets = (unet1, )
        h1, w1 = [int(i) for i in args.model_type.split('_')[1].split('x')]
        image_sizes = ((h1, w1), )
        args.unet_number = 1
    elif args.model_type.startswith('sr'):
        addi_kwargs = dict()
        if not args.condition_on_text:
            addi_kwargs.update(layer_cross_attns=False)
        unet1 = NullUnet()
        unet2 = SRJointUnet(channels_lbl=args.channels_lbl, num_classes=args.num_classes, **addi_kwargs)
        unets = (unet1, unet2)
        h1, w1 = [int(i) for i in args.model_type.split('_')[1].split('x')]
        h2, w2 = [int(i) for i in args.model_type.split('_')[2].split('x')]
        image_sizes = ((h1, w1), (h2, w2))
        args.unet_number = 2
    else:
        raise NotImplementedError(args.model_type)

    # imagen, which contains the unets above (base unet and super resoluting ones)

    imagen = JointImagen(
        unets=unets,
        text_encoder_name='t5-large',
        image_sizes=image_sizes,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        sample_timesteps=args.sample_timesteps,
        cond_drop_prob=args.cond_drop_prob,
        condition_on_text=args.condition_on_text,
        pred_objectives=args.pred_objectives,
        noise_schedules=args.noise_schedules,
        noise_schedules_lbl=args.noise_schedules_lbl,
        cosine_p_lbl=args.cosine_p_lbl,
    )
    trainer = JointImagenTrainer(
        imagen,
        fp16=args.fp16,
        dl_tuple_output_keywords_names=('images', 'labels', 'texts'),
    )
    trainer.load(args.checkpoint_path[0])

    if args.dataset == 'cityscapes':
        mapping_id = id_type_to_classes_cityscapes['train_id']['map_fn']
        transform_lbl = transform_lbl_cityscapes
    elif args.dataset == 'celeba':
        mapping_id = mapping_id_celeba
        transform_lbl = transform_lbl_celeba
    else:
        raise NotImplementedError(args.dataset)

    start_image_or_video, start_label_or_video = None, None

    n_idx = args.start_sample_idx
    while n_idx < args.end_sample_idx:
        assert args.save_path.endswith(('.png', ))
        os.makedirs(osp.dirname(args.save_path), exist_ok=True)

        batch_size = 0
        texts = []
        for test_caption in args.test_captions:
            if n_idx >= args.end_sample_idx:
                break
            texts.append(test_caption)
            batch_size += 1
            n_idx += 1

        # for up- / upup- sampler they need an image and a label for upsample
        if start_at_unet_number > 1:
            start_image_or_video = T.ToTensor()(Image.open(args.start_image_or_video))[None, ...]
            start_label_or_video = ToTensorNoNorm()(Image.open(args.start_label_or_video))[None, ...]
            start_label_or_video = mapping_id[start_label_or_video.long()].float()

        print(f'{n_idx} / {args.end_sample_idx}: {texts}')
        outputs = trainer.sample(
            texts=texts,
            cond_scale=args.cond_scale, batch_size=batch_size,
            start_at_unet_number=start_at_unet_number, stop_at_unet_number=stop_at_unet_number,
            start_image_or_video=start_image_or_video, start_label_or_video=start_label_or_video,
            lowres_sample_noise_level=args.lowres_sample_noise_level,
            return_all_unet_outputs=args.return_all_unet_outputs,
            use_tqdm=True)
        if not args.return_all_unet_outputs:
            outputs = [outputs]
        for idx_unet, output in enumerate(outputs):
            saved_images, saved_labels = output
            saved_labels = transform_lbl(saved_labels, 'train_id')
            saved_grid = [saved_images, saved_labels]
            torchvision.utils.save_image(torch.cat(saved_grid),
                                            args.save_path.replace('.png', f'_{n_idx-1}_{idx_unet}.png'),
                                            nrow=max(2, batch_size), pad_value=1.)
            print(args.save_path.replace('.png', f'_{n_idx-1}_{idx_unet}.png') + ' has been saved.')


if __name__ == '__main__':
    main()
