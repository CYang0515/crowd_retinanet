import json
import csv

'''
generate head/full/visable pair json data  only include head is available gtbox
'''

import numpy as np
import json

def data_gen(root):
    hfv_list = []
    with open(root, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_inf = json.loads(line)
            image_name = img_inf["ID"] + ".jpg"
            gtbox = img_inf["gtboxes"]
            per_hfv_list = [image_name]
            for box in gtbox:
                ignore = box["head_attr"]["ignore"] if box["tag"] == "person" and "ignore" in box["head_attr"] else 1
                head_box = box["hbox"]
                full_box = box["fbox"]
                visble_box = box["vbox"]
                if ignore == 0:
                    per_hfv_list.append(head_box)
                    per_hfv_list.append(full_box)
                    per_hfv_list.append(visble_box)
            hfv_list.append(per_hfv_list)
    return hfv_list

if __name__ == "__main__":
    root_train = "./annotation_train.odgt"
    root_test = "./annotation_val.odgt"
    save_data = "./crowdhuman{}_{}.json"
    train = data_gen(root_train)
    val = data_gen(root_test)
    # for i, [t, v] in enumerate(zip(train, val)):
    with open(save_data.format("train", 'hfv'), 'w') as f:
        json.dump(train, f)
    with open(save_data.format("val", 'hfv'), 'w') as f:
        json.dump(val, f)



