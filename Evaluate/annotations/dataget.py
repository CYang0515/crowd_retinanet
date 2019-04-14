import os
import json
import numpy as np
from sympy import *

class BBOX():
    def __init__(self,x1,y1,x2,y2):
        for i in [x1,y1,x2,y2]:
            assert isinstance(i, float), 'float is not match'
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
    def clip_by_shape(self, shape):
        self.x1 = np.clip(self.x1, 0, shape[1])
        self.x2 = np.clip(self.x2, 0, shape[1])
        self.y1 = np.clip(self.y1, 0, shape[0])
        self.y2 = np.clip(self.y2, 0, shape[0])
        return np.array([[self.x1,self.y1,self.x2,self.y2]])

    def is_box(self):
        return self.w > 0 and self.h > 0

    def area(self):
        return self.w * self.h
    
    @property
    def w(self):
        return self.x2 - self.x1 + 1

    @property
    def h(self):
        return self.y2 - self.y1 + 1

def solve_equ(ori, obj):
    sx = Symbol('sx')
    sy = Symbol('sy')
    dx = Symbol('dx')
    dy = Symbol('dy')
    x1,y1,x2,y2 = obj
    x11,y11,x22,y22 = ori
    out = solve([sx*x1 + dx - x11, sy*y1 + dy - y11, sx*x2 + dx - x22, sy*y2 + dy - y22 ], [sx,sy,dx,dy])
    return out

def load(head_addr, full_addr):
    with open(head_addr, 'rb') as f:
        head = json.load(f)
        annotations = np.array(head['annotations'])
        categories = np.array(head['categories'])
        images = np.array(head['images'])
    with open(full_addr, 'rb') as f:
        body = json.load(f)
        annotations_f = np.array(body['annotations'])
        categories_f = np.array(body['categories'])
        images_f = np.array(body['images'])
    headimg = processann(annotations, categories, images)
    fullimg = processann(annotations_f, categories_f, images_f)
#     head_match = sorted(headimg, key= lambda x : x['match_id'])
#     full_match = sorted(fullimg, key= lambda x : x['match_id'])
    print('finish head,full annotation making...')
    
    for i,hf in enumerate(zip(headimg, fullimg)):
        h = hf[0]
        f = hf[1]
        assert h['match_id'].all() == f['match_id'].all(), 'match_id is fail'
        assert h['id'] == f['id'], 'img is not match'
        outs = []
        for hb,fb in zip(h['bbox'],f['bbox']):
            out = solve_equ(fb, hb)
            outs.append(out)
        headimg[i]['match_d'] = np.array(outs)
        if i%100 ==0:
            print('finish {} /{}...'.format(i, len(headimg)))
        
    return headimg
    
    
    
    
    
        
def processann(annotations, categories, images):      
    imgtoann = []
    idtoanno = np.array([i['image_id'] for i in annotations])
    image = []
    for ids in images:
        inf = {}
        inf['id'] = ids['id']
        inf['file_name'] = '/home/yc/tensorpack_modify/examples/FasterRCNN/COCO/train2014/'+ ids['file_name']
        inf['height'] = ids['height']
        inf['width'] = ids['width']
        anno = np.where(idtoanno == inf['id'])
        anns = annotations[anno]
        for obj in anns:
            if obj['ignore'] == 1 or obj['area'] <=1:
                continue
            x1, y1, w, h = obj['bbox']
                
            box = [float(x1), float(y1),float(x1 + w), float(y1 + h)]
            cbox = BBOX(*box)
            box = cbox.clip_by_shape([inf['height'], inf['width']])
            if not cbox.is_box() or cbox.area() <4:
                continue
            
            if 'bbox' in inf.keys():
                inf['bbox'] = np.concatenate((inf['bbox'], box), axis = 0)
                inf['iscrowd'].append(obj['iscrowd'])
                inf['class'].append(1)
                inf['match_id'].append(obj['id'])
            else:
                inf['bbox'] = box
                inf['iscrowd'] = [obj['iscrowd']]
                inf['class'] = [1]
                inf['match_id'] = [obj['id']]
        try:
            assert 'bbox' in inf.keys(), 'no gt box in '+ inf['filename']
        except:
            continue
            
        inf['bbox'] = np.array(inf['bbox'])
        inf['iscrowd'] = np.array(inf['iscrowd'])
        inf['class'] = np.array(inf['class'])
        inf['match_id'] = np.array(inf['match_id'])
        image.append(inf)
    return np.array(image)

if __name__ == '__main__':
    addrh = './instances_train_head2014.json'
    addrf = './instances_train_visible2014.json'
    with open('./list.json', 'wb') as f:
        lout = load(addrh, addrf)
        json.dump(lout,f)