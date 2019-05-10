import torch
import csv
import numpy as np
from collections import defaultdict
from PIL import ImageDraw, Image
def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a+1], 1)
    return torch.cat([a-b/2,a+b/2], 1)

def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def average_iou(dict_box, thread):
    total_iou = []
    for key, value in dict_box.items():
        box = torch.Tensor(value)
        iou = box_iou(box, box)
        iou = iou.numpy()
        index = np.where(np.logical_and(iou>thread, iou<1))
        sel_value = iou[index]
        total_iou.extend(sel_value)
    return sum(total_iou)/len(total_iou), len(total_iou)

def merge(dict_box, thread):
    merge_box = defaultdict(list)
    for key, value in dict_box.items():
        box = torch.Tensor(value)
        iou = box_iou(box, box)
        iou = iou.numpy()
        n = iou.shape[0]
        angle = np.eye(n) * 2
        mask = np.zeros(n, dtype=bool)
        iou = iou - angle
        while(1):
            if iou.shape[0] == 0:
                break
            index = np.argmax(iou)
            max_value = np.max(iou)
            x = index % n
            y = index // n
            if max_value > thread:
                merge_1 = box[x].numpy()
                merge_2 = box[y].numpy()
                merge_2 = np.stack((merge_1, merge_2))
                x1, x2, y1, y2 = np.min(merge_2[:,0::2]), np.max(merge_2[:,0::2]), np.min(merge_2[:,1::2]), np.max(merge_2[:,1::2])
                merge_box[key].append([x1, y1, x2, y2])
                mask[[x, y]] = True
                iou[mask] = -1
                # np.delete()
                iou[:, mask] = -1
            else:
                mask = np.logical_not(mask)
                no_merge_box = box.numpy()[mask].tolist()
                merge_box[key].extend(no_merge_box)
                break
    return merge_box

def visiual(dict_box, merge_box):
    for key, value in dict_box.items():
        img = Image.open(key)
        draw = ImageDraw.Draw(img)
        for box in value:
            draw.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline='red')
        merge = merge_box[key]
        for box in merge:
            draw.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline='blue')
        img.save('./test.jpg')

if __name__ == '__main__':
    annotations_dir = './pair_val_all.csv'
    out = open(annotations_dir, 'r')
    read = csv.reader(out)
    dict_box = defaultdict(list)
    for i in read:
        name = i[0]
        full = [eval(x) for x in i[5:9]]
        dict_box[name].append(full)

    av, av_num = average_iou(dict_box, 0.5)
    av1_ut = []
    for t in range(50,100,5):
        print(t)
        merge_box = merge(dict_box, t/100)
        av1, av1_num = average_iou(merge_box, 0.5)
        av1_ut.append(av1)
        av1_ut.append(av1_num)
        visiual(dict_box, merge_box)
    p =1


