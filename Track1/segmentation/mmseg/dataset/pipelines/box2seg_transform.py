import mmcv
import numpy as np

from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Crop_with_Mask(object):

    def __init__(self, jitter_ratio=0.1, max_range=.5):
        self.jitter_ratio = jitter_ratio
        self.max_range = max_range

    def get_crop_bbox(self, img):
        if (img > 0).astype(np.float32).sum() == 0:
            return 0, 0, 0, 0
        h,w = img.shape
        x = np.linspace(0, w-1, w).astype(np.float32)
        y = np.linspace(0, h-1, h).astype(np.float32)
        hor, ver = np.meshgrid(x, y)

        min_x = hor[img>0].min().astype(np.int32)
        min_x = max(0, min_x-20)
        min_y = ver[img>0].min().astype(np.int32)
        min_y = max(0, min_y-20)
        max_x = hor[img>0].max().astype(np.int32)
        max_x = min(w, max_x+20)
        max_y = ver[img>0].max().astype(np.int32)
        max_y = min(h, max_y+20)
        return min_x, min_y, max_x, max_y

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def vassign(self, img):
        img[img > 0] = 1
        #img[img == 0] = 255
        #img[img == 1] = 0
        #img[img == 255] = 1
        return img

    def vassign_ignore(self, img):
        img[img > 0] = 255
        img[img == 0] = 255
        return img

    def random_bbox_jitter(self, bbox, height, width):
        #h, w = bbox[3], bbox[2]
        x0, y0, x1, y1 = bbox
        h = y1 - y0
        w = x1 - x0
        if h*w == 0:
            return 0, 0, 0, 0
        g = np.random.normal(0, 1, size=(2))
        g = np.clip(g, -self.max_range, self.max_range)
        x0 += self.jitter_ratio * 0.5 * g[0] * w
        x1 += self.jitter_ratio * 0.5 * g[0] * w
        y0 += self.jitter_ratio * 0.5 * g[1] * h
        y1 += self.jitter_ratio * 0.5 * g[1] * h

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(x1, width)
        y1 = min(y1, height)

        return int(x0), int(y0), int(x1), int(y1)

    def __call__(self, results):

        img = results['img']
        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.vassign(results[key])

        seg = results['gt_semantic_seg']
        crop_bbox = self.get_crop_bbox(seg)
        h,w = img.shape[:2]
        results['ori_box2seg_crop_bbox'] = crop_bbox
        crop_bbox = self.random_bbox_jitter(crop_bbox, h, w)

        if (crop_bbox[2] - crop_bbox[0]) * (crop_bbox[3] - crop_bbox[1]) < 2500:
            crop_bbox = [0, 0, w//2, h//2]
            img = self.crop(img, crop_bbox)
            img_shape = img.shape
            results['img'] = img
            results['img_shape'] = img_shape
            for key in results.get('seg_fields', []):
                results[key] = self.vassign_ignore(results[key])
                results[key] = self.crop(results[key], crop_bbox)
            return results

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['jitter_box2seg_crop_bbox'] = crop_bbox

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(jitter_ratio={self.jitter_ratio})'

@PIPELINES.register_module()
class RandomGamma(object):
    """Using gamma correction to process the image.

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma_range=[80,120]):
        self.gamma_range = gamma_range

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        gamma = np.random.randint(self.gamma_range[0], self.gamma_range[1]) / 100.
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), table)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma_range})'


