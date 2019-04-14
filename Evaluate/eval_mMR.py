from coco import COCO
from eval_MR_multisetup import COCOeval
'''evaluate MR'''
annType = 'bbox'      #specify type here
print('Running demo for *%s* results.'%(annType))

def mMR(resFile, annFile):
    '''
    :param resFile:  json file  detect result : list =[ dict, dict ...] dict = {'image_id':, 'bbox':, 'score':,
    'category_id': }bbox = [x,y,w,h]  image_id = ***(no .jpg)  category_id = 1 for person score must be sort from high to low
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
    resFile = './full.json'
    annFile = './annotations/instances_val_full2014.json'
    mMR(resFile=resFile, annFile=annFile)