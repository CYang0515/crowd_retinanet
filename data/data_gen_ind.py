'''
produce .json file that is needed by data_generate.py
for h, f, v file individual
'''


import numpy as np
import json

def data_gen(root):
    head_list = []
    body_list = []
    visible_list = []
    with open(root, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_inf = json.loads(line)
            image_name = img_inf["ID"] + ".jpg"
            gtbox = img_inf["gtboxes"]
            per_head_list = [image_name]
            per_body_list = [image_name]
            per_visible_list = [image_name]
            for box in gtbox:
                ignore = box["head_attr"]["ignore"] if box["tag"] == "person" and "ignore" in box["head_attr"] else 1
                ignore_full = box["extra"]["ignore"] if "ignore" in box["extra"] else 0
                head_box = box["hbox"]
                visble_box = box["vbox"]
                full_box = box["fbox"]
                head_box = np.append(head_box, 1)
                visble_box = np.append(visble_box, 1)
                full_box = np.append(full_box, 1)
                if ignore_full == 0:
                    per_body_list.extend(full_box.tolist())
                if ignore_full == 0:
                    per_visible_list.extend(visble_box.tolist())
                if ignore == 0:
                    per_head_list.extend(head_box.tolist())
            # if len(per_head_list) > 1:
            head_list.append(per_head_list)
            # if len(per_visible_list) > 1:
            visible_list.append(per_visible_list)
            # if len(per_body_list) > 1:
            body_list.append(per_body_list)
    return head_list, visible_list, body_list

if __name__ == "__main__":
    root_train = "./annotation_train.odgt"
    root_test = "./annotation_val.odgt"
    save_data = "./crowdhuman{}_{}.json"
    name_dict = ["head", "visible", "body"]
    train = data_gen(root_train)
    val = data_gen(root_test)
    for i, [t, v] in enumerate(zip(train, val)):
        with open(save_data.format("train", name_dict[i]), 'w') as f:
            json.dump(t, f)
        with open(save_data.format("val", name_dict[i]), 'w') as f:
            json.dump(v, f)



