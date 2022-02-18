# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, inference_segmentor_w_bbox, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    print('Start Inference')
    bbox = [0,0,100,100]
    result = inference_segmentor_w_bbox(model, args.img, bbox)[0]
    # show the results
    cv2.imwrite('demo.png', result)

if __name__ == '__main__':
    main()
