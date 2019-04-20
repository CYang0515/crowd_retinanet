from __future__ import print_function

import numpy as np
import json
import os

import torch

def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()

    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    # all_detections[index][label] = np.zeros((0, 5))  no modify
                    all_detections[index][label] = np.zeros((0, 9))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def evaluate(
        generator,
        retinanet,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    all_detections = _get_detections(generator, retinanet, score_threshold=score_threshold,
                                     max_detections=max_detections, save_path=save_path)
    image_names = generator.image_names
    full_detects = []
    head_detects = []
    vis_detects = []
    for i, detect in enumerate(all_detections):
        detect = detect[0]
        full_box = detect[:, 0: 4]
        head_box = detect[:, 4: 8]
        full_box[:, 2] = full_box[:, 2] - full_box[:, 0]
        full_box[:, 3] = full_box[:, 3] - full_box[:, 1]
        head_box[:, 2] = head_box[:, 2] - head_box[:, 0]
        head_box[:, 3] = head_box[:, 3] - head_box[:, 1]
        score = detect[:, 8]
        img_id = image_names[i].split('/')[-1][:-4]
        for j in range(len(score)):
            full = {
                'image_id': img_id,
                'bbox': full_box[j].tolist(),
                'score': score[j],
                'category_id': 1,
            }
            head = {
                'image_id': img_id,
                'bbox': head_box[j].tolist(),
                'score': score[j],
                'category_id': 1,
            }
            full_detects.append(full)
            head_detects.append(head)
    with open('./Evaluate/res_full.json', 'w') as f:
        json.dump(full_detects, f)

    with open('./Evaluate/res_head.json', 'w') as f:
        json.dump(head_detects, f)






