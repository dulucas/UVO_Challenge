from __future__ import print_function

import os
import cv2
import time
import glob
import json
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from model import CRW

import utils_videowalk
import utils_videowalk.test_utils as test_utils
from pycocotools import mask as transform_mask

from config import config
from raft import RAFT

######################################################################

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize( img, (owidth, oheight) )
    img = im_to_torch(img)
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:,:,::-1]
    img = img.copy()
    return im_to_torch(img)

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

######################################################################

def try_np_load(p):
    try:
        return np.load(p)
    except:
        return None

def make_lbl_set(lbls):
    lbl_set = [np.zeros(3).astype(np.uint8)]
    count_lbls = [0]

    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)

    return lbl_set

def texturize(onehot):
    flat_onehot = onehot.reshape(-1, onehot.shape[-1])
    lbl_set = np.unique(flat_onehot, axis=0)

    count_lbls = [np.all(flat_onehot == ll, axis=-1).sum() for ll in lbl_set]
    object_id = np.argsort(count_lbls)[::-1][1]

    hidxs = []
    for h in range(onehot.shape[0]):
        appears = np.any(onehot[h, :, 1:] == 1)
        if appears:
            hidxs.append(h)

    nstripes = min(10, len(hidxs))

    out = np.zeros((*onehot.shape[:2], nstripes+1))
    out[:, :, 0] = 1

    for i, h in enumerate(hidxs):
        cidx = int(i // (len(hidxs) / nstripes))
        w = np.any(onehot[h, :, 1:] == 1, axis=-1)
        out[h][w] = 0
        out[h][w, cidx+1] = 1

    return out

def context_index_bank(n_context, long_mem, N):
    '''
    Construct bank of source frames indices, for each target frame
    '''
    ll = []   # "long term" context (i.e. first frame)
    for t in long_mem:
        assert 0 <= t < N, 'context frame out of bounds'
        idx = torch.zeros(N, 1).long()
        if t > 0:
            idx += t + (n_context+1)
            idx[:n_context+t+1] = 0
        ll.append(idx)
    # "short" context
    ss = [(torch.arange(n_context)[None].repeat(N, 1) +  torch.arange(N)[:, None])[:, :]]
    ll = [ss[0] - 1]
    ll[0][ll[0] < 0] = 0

    return ll + ss

def iou(sm, dm):
    return np.logical_and(sm, dm).astype(np.float32).sum() / (np.logical_or(sm, dm).astype(np.float32).sum() + 1e-6)

def mask_matching(smasks, dmasks, thre=0.5):
    matching_matrix = np.zeros((len(smasks), len(dmasks)))
    for i in range(len(smasks)):
        for j in range(len(dmasks)):
            matching_matrix[i,j] = iou(smasks[i], dmasks[j])

    matched_indexes = np.arange(len(smasks))[matching_matrix.max(1) > thre]
    matched_tgt_indexes = np.array([matching_matrix[i].argmax() for i in matched_indexes])

    no_matched_indexes = set(np.arange(len(smasks)).tolist()) - set(matched_indexes.tolist())
    no_matched_indexes = np.array(list(no_matched_indexes))

    new_instances_indexes = set(np.arange(len(dmasks)).tolist()) - set(matched_tgt_indexes.tolist())
    new_instances_indexes = np.array(list(new_instances_indexes))
    return matched_indexes, matched_tgt_indexes, no_matched_indexes, new_instances_indexes

def mask_nms(ms, num_masks, thre=.5):
    rest_indexes = np.arange(len(ms)).tolist()
    keep_indexes = []
    while len(rest_indexes) > 0 and len(keep_indexes) < num_masks:
        last = 0
        cur = ms[rest_indexes[last]]
        keep_indexes.append(rest_indexes[last])
        del rest_indexes[last]

        dlt_set = []
        for i in rest_indexes:
            cha = ms[i]
            iou = np.logical_and(cur, cha).astype(np.float32).sum() / (np.logical_or(cur, cha).astype(np.float32).sum() + 1e-6)
            if iou > thre:
                dlt_set.append(i)

        for i in dlt_set:
            rest_indexes.remove(i)
    return keep_indexes

def generate_opflow(model, image_prev, image_nex, num_iters=20):
    image_prev = torch.tensor(image_prev).cuda().permute(2, 0, 1).unsqueeze(0)
    image_nex = torch.tensor(image_nex).cuda().permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        _, flow_forward = model(image_prev, image_nex, iters=num_iters, test_mode=True)
    flow_forward = flow_forward.squeeze()
    flow_forward = flow_forward.permute(1, 2, 0).cpu().numpy()
    return flow_forward

def warp_flow(img, flow, binarize=True):
    h,w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    if binarize:
        res = np.equal(res, 1).astype(np.uint8)
    return res

######################################################################

class CreateTrackers(object):
    def __init__(self, real_masks, scores, n_context=0):
        self.init_trackers(real_masks, scores)

    def init_trackers(self, real_masks, scores, n_context=0):
        self.trackers = []
        self.dead_trackers = []
        for i in range(len(real_masks[0])):
            tracker = Tracker(real_masks[0][i], scores[0][i], 0, 1)
            self.trackers.append(tracker)

    def add_tracker(self, real_mask, score, start_idx, end_idx):
        # mask shape: H*W
        tracker = Tracker(real_mask, score, start_idx, end_idx)
        self.trackers.append(tracker)

class Tracker(object):
    def __init__(self, real_mask, score, start_idx, end_idx):
        self.alive = True
        self.dead_count = 0
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.real_masks = [real_mask]
        self.score = score

    def kill(self):
        self.alive = False

    def update(self, real_mask, score, idx):
        self.real_masks.append(real_mask)
        self.score += score

class SingleVideo(object):
    def __init__(self, images_path, masks_path, mask_per_frame, nms_thre, mask_matching_thre, patience):
        #self.images_path = glob.glob(images_path + '/*')
        #self.images_path.sort()
        masks_path_ = glob.glob(masks_path + '/*')
        masks_ids = [int(i.split('/')[-1].split('.')[0]) for i in masks_path_]
        masks_ids.sort()
        self.masks_path = [masks_path + '/' + str(i) + '.json' for i in masks_ids]
        self.images_path = [images_path + '/' + str(i) + '.png' for i in masks_ids]
        self.mask_per_frame = mask_per_frame
        self.nms_thre = nms_thre
        self.mask_matching_thre = mask_matching_thre
        self.patience = patience

        #self.images_path = self.images_path[:10]
        #self.masks_path = self.masks_path[:10]
        model = torch.nn.DataParallel(RAFT(config))
        model.load_state_dict(torch.load(config.opflow_model_path))
        model = model.module
        model.cuda()
        model.eval()
        self.flow_model = model

        self.load_imgs()
        self.load_flow()
        self.load_masks()
        self.load_model()
        assert len(self.imgs) == len(self.masks)
        assert len(self.flows) == len(self.imgs) - 1

    def load_imgs(self):
        self.imgs = []
        for i in self.images_path:
            self.imgs.append(cv2.imread(i))
        print(len(self.imgs), ' Images Loaded')

    def load_masks(self):
        self.masks = []
        self.scores = []
        for ins_path in tqdm(self.masks_path):
            masks = []
            scores = []
            f = json.load(open(ins_path, 'r'))
            f = sorted(f, key=lambda x:-x['score'])
            for i in f:
                mask = transform_mask.decode(i['segmentation'])
                score = i['score']
                masks.append(mask)
                scores.append(score)
            masks = np.array(masks)
            keep_indexes = mask_nms(masks, self.mask_per_frame, self.nms_thre)
            masks = [masks[i] for i in keep_indexes][:self.mask_per_frame]
            scores = [scores[i] for i in keep_indexes][:self.mask_per_frame]
            self.masks.append(masks)
            self.scores.append(scores)
        print(len(self.masks), ' Masks Loaded')

    def load_flow(self):
        assert self.imgs is not None
        self.flows = []
        h,w,_ = self.imgs[0].shape
        rsz_h,rsz_w = h//32*32, w//32*32
        for i in tqdm(range(len(self.imgs)-1)):
            img_prev = cv2.resize(self.imgs[i], (rsz_w,rsz_h))
            img_nex = cv2.resize(self.imgs[i+1], (rsz_w,rsz_h))
            flow = generate_opflow(self.flow_model, img_prev, img_nex, 20)
            flow[:,:,0] = flow[:,:,0] / rsz_w
            flow[:,:,1] = flow[:,:,1] / rsz_h
            flow = cv2.resize(flow, (w,h))
            flow[:,:,0] *= w
            flow[:,:,1] *= h
            self.flows.append(flow)
        print(len(self.flows), 'Flows Generated')

    #def load_masks(self):
    #    self.masks = []
    #    for ins_path in self.masks_path:
    #        masks = []
    #        mask_ens = Image.open(ins_path)
    #        mask_ens = np.atleast_3d(mask_ens)[...,0]
    #        for i in np.unique(mask_ens):
    #            if i == 0:
    #                continue
    #            masks.append((mask_ens == i).astype(np.float32))
    #        self.masks.append(masks)
    #    print(len(self.masks), ' Masks Loaded')

    def load_model(self):
        self.args = utils_videowalk.arguments.test_args()

        self.args.imgSize = self.args.cropSize
        print('Context Length:', self.args.videoLen, 'Image Size:', self.args.imgSize)
        print('Arguments', self.args)

        self.model = CRW(self.args, vis=None).to(self.args.device)
        self.args.mapScale = test_utils.infer_downscale(self.model)

        self.args.use_lab = self.args.model_type == 'uvc'

        print('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))

        # Load checkpoint.
        if os.path.isfile(self.args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(self.args.resume)

            if self.args.model_type == 'scratch':
                state = {}
                for k,v in checkpoint['model'].items():
                    if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
                        state[k.replace('.1.weight', '.weight')] = v
                    else:
                        state[k] = v
                utils_videowalk.partial_load(state, self.model, skip_keys=['head'])
            else:
                utils_videowalk.partial_load(checkpoint['model'], self.model, skip_keys=['head'])

            del checkpoint

        self.model.eval()
        self.model = self.model.to(self.args.device)

    def remove_duplicate_trackers(self, t):
        self.trackers.trackers = sorted(self.trackers.trackers, key = lambda x:(x.start_idx - x.end_idx, -x.score))
        trackers = self.trackers.trackers
        indexes = [i for i in range(len(trackers)) if trackers[i].dead_count == 0]
        trackers_masks = [trackers[i].real_masks[-1] for i in indexes]
        trackers_masks = np.array(trackers_masks)
        keep_indexes = mask_nms(trackers_masks, len(trackers_masks)+1, self.nms_thre)
        keep_tracker_indexes = [indexes[i] for i in keep_indexes]
        remove_tracker_indexes = list(set(indexes) - set(keep_tracker_indexes))
        dead_indexes = []
        for i in remove_tracker_indexes:
            self.trackers.trackers[i].kill()
            self.trackers.trackers[i].end_idx = t - 1 # Attention, here we also remove the last frame of this tracker, because it coincide with previous tracker
            self.trackers.dead_trackers.append(self.trackers.trackers[i])
            dead_indexes.append(i)
        dead_indexes.sort()
        dead_indexes.reverse()
        for dead_index in dead_indexes:
            self.trackers.trackers.pop(dead_index)

    def inference(self):

        ori_h, ori_w, _ = self.imgs[0].shape

        n_context = self.args.videoLen
        assert n_context == 0

        print('******* (%s frames) *******' % (len(self.imgs)))

        ##################################################################
        # Propagate Labels and Save Predictions
        ###################################################################
        self.trackers = CreateTrackers(self.masks, self.scores)

        for t in tqdm(range(1, len(self.imgs))):
            propogated_masks = []
            propogated_preds = []
            for tracker in self.trackers.trackers:
                prev_mask = tracker.real_masks[-1]
                pred = warp_flow(prev_mask, self.flows[t-1])
                propogated_masks.append(pred)

            detected_masks = self.masks[t]
            detected_scores = self.scores[t]
            propogated_masks = np.array(propogated_masks)
            detected_masks = np.array(detected_masks)
            mtch_indexes, mtch_tgt_indexes, no_mtch_indexes, new_ins_indexes = mask_matching(propogated_masks, detected_masks, self.mask_matching_thre)

            for src, tgt in zip(mtch_indexes, mtch_tgt_indexes):
                self.trackers.trackers[src].update(detected_masks[tgt], detected_scores[tgt], t)
                self.trackers.trackers[src].dead_count = 0
                self.trackers.trackers[src].end_idx = t+1

            dead_indexes = []
            for no_mtch in no_mtch_indexes:
                self.trackers.trackers[no_mtch].dead_count += 1
                if self.trackers.trackers[no_mtch].dead_count > self.patience:
                    self.trackers.trackers[no_mtch].kill()
                    self.trackers.trackers[no_mtch].end_idx = t - self.patience
                    self.trackers.dead_trackers.append(self.trackers.trackers[no_mtch])
                    dead_indexes.append(no_mtch)
                else:
                    self.trackers.trackers[no_mtch].update(propogated_masks[no_mtch], 0, t)

            dead_indexes.sort()
            dead_indexes.reverse()
            for dead_index in dead_indexes:
                self.trackers.trackers.pop(dead_index)

            self.remove_duplicate_trackers(t)

            for new_ins in new_ins_indexes:
                if detected_masks[new_ins].sum() == 0:
                    continue
                self.trackers.add_tracker(detected_masks[new_ins], detected_scores[new_ins], t, t+1)

    def format_results(self, save_path, vid):
        assert self.trackers is not None
        if not os.path.exists(save_path + '/' + vid):
            os.makedirs(save_path + '/' + vid)
        for index, tracker in enumerate(self.trackers.trackers):
            for i in range(tracker.start_idx, tracker.end_idx):
                img = self.imgs[i]
                mask = tracker.real_masks[i - tracker.start_idx]
                img[:,:,-1] = mask.astype(np.uint8)*255
                cv2.imwrite(save_path + '/' + vid + '/tracker_' + str(index).zfill(4) + '_img' + str(i).zfill(3) + '.jpg', img)
        for index, tracker in enumerate(self.trackers.dead_trackers):
            for i in range(tracker.start_idx, tracker.end_idx):
                img = self.imgs[i]
                mask = tracker.real_masks[i - tracker.start_idx]
                img[:,:,-1] = mask.astype(np.uint8)*255
                cv2.imwrite(save_path + '/' + vid + '/dead_tracker_' + str(index).zfill(4) + '_img' + str(i).zfill(3) + '.jpg', img)
        #for index in range(len(self.masks)):
        #    for i in range(len(self.masks[index])):
        #        mask = self.masks[index][i].astype(np.uint8)*255
        #        cv2.imwrite(save_path + '/' + str(index).zfill(4) + '_mask' + str(i).zfill(4) + '.jpg', mask)

    def format_submissions(self, vid, len_thre=30):
        assert self.trackers is not None
        output = []
        trackers = self.trackers.trackers + self.trackers.dead_trackers
        for index, tracker in enumerate(trackers):
            if tracker.end_idx - tracker.start_idx < len_thre:
                continue
            ins = dict()
            ins['video_id'] = vid
            ins['score'] = (tracker.end_idx - tracker.start_idx) * tracker.score
            ins['category_id'] = 1
            segmentations = [None for i in range(tracker.start_idx)]
            for i in range(tracker.start_idx, tracker.end_idx):
                mask = tracker.real_masks[i - tracker.start_idx]
                segmentation = transform_mask.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                segmentation['counts'] = segmentation['counts'].decode("utf-8")
                segmentations.append(segmentation)
            for i in range(90 - tracker.end_idx):
                segmentations.append(None)
            assert len(segmentations) == 90
            ins['segmentations'] = segmentations
            output.append(ins)
        return output



if __name__ == '__main__':

    vids = json.load(open('PATH/TO/YOUR/UVO_video_test_dense.json', 'r'))['videos']
    name2id = dict()
    for i in vids:
        name2id[i['ytid']] = i['id']

    output = []
    for vid in vids:
        print(vid)
        single = SingleVideo('PATH/TO/YOUR/UVO_Dense_frames/' + vid + '/',
                            'PATH/TO/YOUR/UVO_Prediction_for_frames/' + vid + '/',
                            100,
                            nms_thre=0.7,
                            mask_matching_thre=0.5,
                            patience=5)
        single.inference()
        ins = single.format_submissions(name2id[vid], 5)
        output.extend(ins)
        del single
        torch.cuda.empty_cache()

    with open('./subms.json', 'w') as w:
        json.dump(output, w)


