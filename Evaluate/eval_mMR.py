from coco import COCO
from eval_MR_multisetup import COCOeval
from collections import defaultdict
import json
'''evaluate MR'''
annType = 'bbox'      #specify type here
print('Running demo for *%s* results.'%(annType))
annFile = '../COCO/annotations/instances_val_{}2014.json'.format('body')

def mMR(resFile, annFile):
    '''
    :param resFile:  json file  detect result : list =[ dict, dict ...] dict = {'image_id':, 'bbox':, 'score':,
    'category_id': }bbox = [x,t,w,h]  image_id = ***(no .jpg)  category_id = 1 for person score must be sort from high to low
    :param annFile:  json file  format is same as mscoco dataset for example instances_val_{}2014.json
    :return: None
    '''
    res_file = open("results.txt", "w")
    for id_setup in range(3, 4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        cocoEval.summarize(id_setup, res_file)
    res_file.close()

if __name__ == '__main__':
    resFile = './resFile.json'
    annFile = '/home/yc/tensorpack_modify/examples/FasterRCNN/COCO/annotations/instances_val_full2014.json'
    # outcomes = './outcomes.json'
    # '''when use retinanet output for the first time''' if change retinanet output ,the code is not needed.
    # with open(outcomes, 'r') as f:
    #     ff = json.load(f)
    #     for j in ff:
    #         j['image_id'] = j['image_id'][0][0:-4]
    #         j['category_id'] = j['category']
    #         del j['category']
    # with open(resFile, 'w') as f:
    #     json.dump(ff, f)
    #
    # with open(annFile) as f:
    #     fi = json.load(f)

    # view outcome
    with open(resFile) as f:
        file = json.load(f)
    root = '/home/yc/tensorpack_modify/examples/FasterRCNN/COCO/val2014/'
    from PIL import Image, ImageDraw
    from collections import defaultdict

    imgtodt = defaultdict(list)
    for i in file:
        image_id = i['image_id']
        bbox = i['bbox']
        score = i['score']
        bbox.append(score)
        imgtodt[image_id].append(bbox)
    for image_id, bboxes in imgtodt.items():
        img = Image.open(root + image_id + '.jpg')
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            if bbox[4] > 0.5:
                draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], outline='red')
        img.save('./test.jpg')

    mMR(resFile=resFile, annFile=annFile)