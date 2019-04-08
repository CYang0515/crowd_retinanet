'''
generate union hfv csv file  format Imagename, (h)x,y,w,h, (f)x,y,w,h, (v)x,y,w,h  cat for one person(len = 13)

include full is visible , no matter whether head is visible
'''

import json
import csv
import numpy as np

train_data = '/home/YC/FasterRCNN/COCO/train2014/'
val_data = '/home/YC/FasterRCNN/COCO/val2014/'
body_train_csv = './pair_train_all.csv'
body_train_json = './crowdhumantrain_hfv_all.json'
body_val_csv = './pair_val_all.csv'
body_val_json = './crowdhumanval_hfv_all.json'

def ann_csv():
    out = open(body_train_csv, 'w', newline='')
    csv_write = csv.writer(out)
    with open(body_train_json) as f:
        file = json.load(f)
        for line in file:
            image_name = train_data + line[0]
            head_igore = line[4::4]
            del line[4::4]
            bbox = np.array(line[1:]).reshape([-1, 3, 4])
            for ids, box in enumerate(bbox):
                box[:, 2] = box[:, 2] + box[:, 0]
                box[:, 3] = box[:, 3] + box[:, 1]
                box = box.reshape([-1]).tolist()
                box.append('person')
                anno = [image_name]
                anno.extend(box)
                anno.extend([head_igore[ids]])
                csv_write.writerow(anno)
    out.close()

    out = open(body_val_csv, 'w', newline='')
    csv_write = csv.writer(out)
    with open(body_val_json) as f:
        file = json.load(f)
        for line in file:
            image_name = val_data + line[0]
            head_igore = line[4::4]
            del line[4::4]
            bbox = np.array(line[1:]).reshape([-1, 3, 4])
            for ids, box in enumerate(bbox):
                box[:, 2] = box[:, 2] + box[:, 0]
                box[:, 3] = box[:, 3] + box[:, 1]
                box = box.reshape([-1]).tolist()
                box.append('person')
                anno = [image_name]
                anno.extend(box)
                anno.extend([head_igore[ids]])
                csv_write.writerow(anno)
    out.close()
# ann_csv()
out = open(body_val_csv, 'r')
read = csv.reader(out)
from PIL import ImageDraw, Image
for x, line in enumerate(read):
    name = line[0]
    box = line[1:-2]
    if int(line[-1]) == 1:
        ori = name
        img = Image.open(name)
        draw = ImageDraw.Draw(img)
    # if x == 0:
    #     ori = name
    #     img = Image.open(name)
    #     draw = ImageDraw.Draw(img)
    # else:
    #     if name != ori:
    #         break
        draw.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline='red')
        draw.rectangle([float(box[4]), float(box[5]), float(box[6]), float(box[7])], outline='green')
        draw.rectangle([float(box[8]), float(box[9]), float(box[10]), float(box[11])], outline='blue')
        # break
        img.save('./test.jpg')