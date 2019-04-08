'''
generate individual head full visible csv file
'''

body_train_json = './crowdhumantrain_body.json'
body_val_json = './crowdhumanval_body.json'
head_train_json = './crowdhumantrain_head.json'
head_val_json = './crowdhumanval_head.json'
vis_train_json = './crowdhumantrain_visible.json'
vis_val_json = './crowdhumanval_visible.json'
train_data = '/home/YC/FasterRCNN/COCO/train2014/'  # file dir that storage image data
val_data = '/home/YC/FasterRCNN/COCO/val2014/'

body_train_csv = './train_data.csv'
body_val_csv = './val_data.csv'
head_train_csv = './head_train_data.csv'
head_val_csv = './head_val_data.csv'
vis_train_csv = './vis_train_data.csv'
vis_val_csv = './vis_val_data.csv'
class_csv = './class.csv'

which = 3

if which == 2:
    body_train_json = head_train_json
    body_val_json = head_val_json
    body_train_csv = head_train_csv
    body_val_csv = head_val_csv
elif which == 3:
    body_train_json = vis_train_json
    body_val_json = vis_val_json
    body_train_csv = vis_train_csv
    body_val_csv = vis_val_csv


import json
import csv
import numpy as np

out = open(class_csv, 'w', newline='')
csv_write = csv.writer(out)
csv_write.writerow(['person', 0])
out.close()

def ann_csv():
    out = open(body_train_csv, 'w', newline='')
    csv_write = csv.writer(out)
    with open(body_train_json) as f:
        file = json.load(f)
        for line in file:
            image_name = train_data + line[0]
            bbox = np.array(line[1:]).reshape([-1, 5])
            for box in bbox:
                box = list(box[:-1])
                box[2] = box[2] + box[0]
                box[3] = box[3] + box[1]
                box.append('person')
                anno = [image_name]
                anno.extend(box)
                csv_write.writerow(anno)
    out.close()

    out = open(body_val_csv, 'w', newline='')
    csv_write = csv.writer(out)
    with open(body_val_json) as f:
        file = json.load(f)
        for line in file:
            image_name = val_data + line[0]
            bbox = np.array(line[1:]).reshape([-1, 5])
            for box in bbox:
                box = list(box[:-1])
                box[2] = box[2] + box[0]
                box[3] = box[3] + box[1]
                box.append('person')
                anno = [image_name]
                anno.extend(box)
                csv_write.writerow(anno)
    out.close()
ann_csv()
out = open(body_val_csv, 'r')
read = csv.reader(out)
from PIL import ImageDraw, Image  # vis data whether is true
for x, line in enumerate(read):
    name = line[0]
    box = line[1:-1]
    if x == 0:
        ori = name
        img = Image.open(name)
        draw = ImageDraw.Draw(img)
    else:
        if name != ori:
            break
    draw.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline='red')
img.save('./test.jpg')