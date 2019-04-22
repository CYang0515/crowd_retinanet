# -*- coding: utf-8 -*-
# File: eval.py

import os
from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from recocoeval import RCOCOeval as COCOeval
# DetectionResult = namedtuple(
#     'DetectionResult',
#     ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""
# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb

class params():
    def __init__(self):
        self.BASEDIR = './'
        self.MODE_MASK = False
        self.TYPE = 'full'

def print_evaluation_scores(json_file, name='head'):
    ret = {}
    assert cfg.BASEDIR and os.path.isdir(cfg.BASEDIR)
    if name == 'head':
        annofile = os.path.join(
            cfg.BASEDIR, 'annotations',
            'instances_val_head2014.json')
    else:
        annofile = os.path.join(
            cfg.BASEDIR, 'annotations',
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
    cfg = params()
    json_file_body = './res_full.json'
    print_evaluation_scores(json_file_body, cfg.TYPE)