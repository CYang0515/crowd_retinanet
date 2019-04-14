import numpy as np
import json
with open('./instances_train_head2014.json') as f:
    head = json.load(f)
with open('./instances_train_visible2014.json') as f:
    visible = json.load(f)
with open('./instances_train_full2014.json') as f:
    full = json.load(f)

image_head = head['images']
image_visible = visible['images']
image_full = full['images']

annotation_head = head['annotations']
annotation_visible = visible['annotations']
annotation_full = full['annotations']
ratio_hv = []
ratio_hf = []
ratio_fwh = []
for i, [h, v, f] in enumerate(zip(annotation_head,annotation_visible,annotation_full)):
    assert h['image_id'] == v['image_id'] and h['image_id'] == f['image_id'], 'image id is fail'
    assert h['id'] == v['id'] and h['id'] == f['id'], 'annotations id is fail'
    if h['ignore'] == 0:#and h['bbox'][2]*h['bbox'][3] <= 512*512 and h['bbox'][2]*h['bbox'][3] >= 64*64:
        hbox = h['bbox']
        vbox = v['bbox']
        fbox = f['bbox']
        khv = hbox[3] / vbox[3]
        khf = hbox[3] / fbox[3]
        fwh = fbox[2] / fbox[3]
        ratio_hf.append(khf)
        ratio_hv.append(khv)
        ratio_fwh.append(fwh)
histogram_hv = np.histogram(np.array(ratio_hv), 3)
histogram_hf = np.histogram(np.array(ratio_hf), 2000)
histogram_fwh = np.histogram(np.array(ratio_fwh), 2000)
mean_hv = np.mean(ratio_hv)
mean_hf = np.mean(ratio_hf)
mean_fwh = np.mean(ratio_fwh)
variance_hv = np.var(ratio_hv)
variance_hf = np.var(ratio_hf)
variance_fwh = np.var(ratio_fwh)
special_hf = np.histogram(ratio_hf, [0, 0.15, 0.17, 1])
special_hv = np.histogram(ratio_hv, [0, 0.245, 0.255, 1])
print(histogram_hf)
print(histogram_hv)
print(mean_hf, variance_hf)
print(mean_hv, variance_hv)
print(special_hf)
print(special_hv)
union = np.arange(0,1,1./len(ratio_hf))
mean_union = np.mean(union)
var_union = np.var(union)
print(mean_union, var_union)
import matplotlib.pyplot as plt
plt.plot(histogram_fwh[0])
plt.show()
p = 1