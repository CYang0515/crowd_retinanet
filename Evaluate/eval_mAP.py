# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
from contextlib import ExitStack
import numpy as np
import cv2

from tensorpack.utils.utils import get_tqdm_kwargs

from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from recocoeval import RCOCOeval as COCOeval
import pycocotools.mask as cocomask

from coco import COCOMeta
from common import CustomResize, clip_boxes
from config import config as cfg

DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


def fill_full_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret


def detect_one_image(img, model_func, name='head'):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """
    if cfg.DATA.WGT:
        img, wrap2, wrap3, wrap4, wrap5, wrap6, orig_shape = img
        # orig_shape = img.shape[:2]
        # resizer = CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
        # resized_img = resizer.augment(img)
        resized_img = img #resize is done in geting eval dataflow
        scale = np.sqrt(resized_img.shape[0] * 1.0 / orig_shape[0] * resized_img.shape[1] / orig_shape[1])
        #     boxes, probs, labels, *masks = model_func(resized_img)
        resized_img = np.expand_dims(resized_img, axis=0)
        wrap2 = np.expand_dims(wrap2, axis=0)
        wrap3 = np.expand_dims(wrap3, axis=0)
        wrap4 = np.expand_dims(wrap4, axis=0)
        wrap5 = np.expand_dims(wrap5, axis=0)
        wrap6 = np.expand_dims(wrap6, axis=0)
        boxes, probs = model_func(resized_img, wrap2, wrap3, wrap4, wrap5, wrap6)
        labels = np.ones((probs.shape[0],))
        boxes = boxes / scale
        # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
        boxes = clip_boxes(boxes, orig_shape)
        print('use gt wrap')
    else:
        orig_shape = img.shape[:2]
        resizer = CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
        resized_img = resizer.augment(img)
        scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    #     boxes, probs, labels, *masks = model_func(resized_img)
        resized_img = np.expand_dims(resized_img, axis=0)
        boxes, probs, boxes_body, probs_body = model_func(resized_img)
        labels = np.ones((probs.shape[0],))
        boxes = boxes / scale

        labels_body = np.ones((probs_body.shape[0],))
        boxes_body = boxes_body / scale
        # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
        boxes = clip_boxes(boxes, orig_shape)

        boxes_body = clip_boxes(boxes_body, orig_shape)

#     if masks:
#     if 0:
#         # has mask
#         full_masks = [fill_full_mask(box, mask, orig_shape)
#                       for box, mask in zip(boxes, masks[0])]
#         masks = full_masks
#     else:
        # fill with none
    masks = [None] * len(boxes)
    masks_body = [None] * len(boxes_body)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    results_body = [DetectionResult(*args) for args in zip(boxes_body, probs_body, labels_body, masks_body)]
    if name == 'head':
        return results
    else:
        return results_body


def eval_coco(df, detect_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = []
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        if not cfg.DATA.WGT:
            for img, img_id in df.get_data():
                results = detect_func(img)
                for r in results:
                    box = r.box
                    cat_id = COCOMeta.class_id_to_category_id[r.class_id]
                    box[2] -= box[0]
                    box[3] -= box[1]

                    res = {
                        'image_id': img_id,
                        'category_id': cat_id,
                        'bbox': list(map(lambda x: round(float(x), 2), box)),
                        'score': round(float(r.score), 3),
                    }

                    # also append segmentation to results
                    if r.mask is not None:
                        rle = cocomask.encode(
                            np.array(r.mask[:, :, None], order='F'))[0]
                        rle['counts'] = rle['counts'].decode('ascii')
                        res['segmentation'] = rle
                    all_results.append(res)
                tqdm_bar.update(1)
        else:
            for img in df.get_data():
                results = detect_func(img[:-1])
                for r in results:
                    box = r.box
                    cat_id = COCOMeta.class_id_to_category_id[r.class_id]
                    box[2] -= box[0]
                    box[3] -= box[1]

                    res = {
                        'image_id': img[-1],
                        'category_id': cat_id,
                        'bbox': list(map(lambda x: round(float(x), 2), box)),
                        'score': round(float(r.score), 3),
                    }
                    all_results.append(res)
                tqdm_bar.update(1)
            print('use gt wrap')
    return all_results


# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores(json_file, name='head'):
    ret = {}
    assert cfg.DATA.BASEDIR and os.path.isdir(cfg.DATA.BASEDIR)
    if name == 'head':
        annofile = os.path.join(
            cfg.DATA.BASEDIR, 'annotations',
            'instances_val_head2014.json')
     #   annofile = os.path.join(
     #       cfg.DATA.BASEDIR, 'annotations',
     #       'instances_{}.json'.format(cfg.DATA.VAL))
    else:
        annofile = os.path.join(
            cfg.DATA.BASEDIR, 'annotations',
            'instances_val_full2014.json')
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    for k in range(6):
        ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

    if cfg.MODE_MASK:
        cocoEval = COCOeval(coco, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        for k in range(6):
            ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
    return ret

if __name__ == '__main__':
    json_file_body = './evaluatemMR/new_0.5nms.json'
    print_evaluation_scores(json_file_body, 'body')