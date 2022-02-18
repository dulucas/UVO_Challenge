# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.001, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    import glob
    from tqdm import tqdm
    import numpy as np
    import json
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test images
    pathes = glob.glob('/IMAGE/PATH/*')
    out = []
    for index, path in enumerate(pathes):
        result = inference_detector(model, path)
        # write the results in COCO format
        for box in result:
            info = dict()
            score = float(box[4])
            box = [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])]
            info['image_id'] = int(index)
            info['category_id'] = int(1)
            info['bbox'] = box
            info['score'] = score
            info['file_name'] = path
            out.append(info)
        with open('demo.json', 'w') as w:
            json.dump(out, w)

async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
