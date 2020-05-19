import os
import numpy as np
import torch
from PIL import Image
from data import MyDataset
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.models.detection import FGFAFasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import random
import backbone
import glob
from torchvision import datasets, transforms
from engine import evaluate
from engine import guide_images_processing
import utils
import argparse
from matplotlib import cm
import copy

def make_boxes(img, boxes, IDs):
    if len(boxes) == 0:
        return img

    line_width = 4
    max_h, max_w = img.shape[:2]

    for idx, box in enumerate(boxes):
        ID = IDs[idx]
        if ID == 1:
            R = 1
            G = 1
            B = 0
        elif ID == 2:
            R = 0
            G = 1
            B = 0
        elif ID == 3:
            R = 1
            G = 0
            B = 1
        elif ID == 4:
            R = 0
            G = 1
            B = 1
        else:
            R = 1
            G = 0.5
            B = 0.5

        for w in range(int(box[0]), int(box[2])+4):
            if w>= max_w:
                break
            hs = [int(box[1]), int(box[3])]
            for h in hs:
                for i in range(line_width):
                    if h+i >= max_h:
                        if h == max_h:
                            i+=1
                        img[h-i][w][0] = R
                        img[h-i][w][1] = G
                        img[h-i][w][2] = B
                    else:
                        img[h+i][w][0] = R
                        img[h+i][w][1] = G
                        img[h+i][w][2] = B
        for h in range(int(box[1]), int(box[3])+4):
            if h >= max_h:
                break
            ws = [int(box[0]), int(box[2])]
            for w in ws:
                for i in range(line_width):
                    if w+i >= max_w:
                        if w == max_w:
                            i+=1
                        img[h][w-i][0] = R
                        img[h][w-i][1] = G
                        img[h][w-i][2] = B
                    else:
                        img[h][w+i][0] = R
                        img[h][w+i][1] = G
                        img[h][w+i][2] = B

    return img

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='settings for training.')
    
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--video', type=str, default='01', help='choose among 01, 40, 45, 47, 49')
    parser.add_argument('--num_batches', type=int, default=1)
    args = parser.parse_args()
    
    print('args: ',args)

    if args.checkpoint is None:
        raise AttributeError('Checkpoint is not defined.')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 7

    backbone = backbone.FGFA_Backbone()
    backbone.out_channels = 1024
    anchor_sizes = ((16, 32, 64, 128),)
    aspect_ratios = ((0.5, 1.0, 2.0),)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names='0',
                                                    output_size=7,
                                                    sampling_ratio=2)
    # move model to the right device
    model = FGFAFasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler,
                   rpn_pre_nms_top_n_train=12000,
                   rpn_pre_nms_top_n_test=6000)
   
    print('load model parameters...')
    model.load_state_dict(torch.load('output/output_epoch_{}.pt'.format(args.checkpoint)))
   
    model.to('cuda')
    model.eval()

    target_video = '../drone/drone-dataset/t1_video/val'
    
    print('Start Visualization')
            
    val_dataset = MyDataset(target_video, Is_training=False)
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.num_batches, shuffle=True, num_workers=1, collate_fn=utils.collate_fn)
    
    data_iter = iter(data_loader)
    
    with torch.no_grad():
        j = 0
        for data in data_iter:
            if j==20:
                break
            ref_image, guide_image, ann= data
            ref_image = list(image.to(device) for image in ref_image)
            targets = [{k: v.to(device) for k, v in t.items()} for t in ann]
            guide_image = guide_images_processing(guide_image, device)
            #print(ann)
            #input()
            #print(ref_image)
            #input()
            #print(guide_image)
            #input()
            output = model(ref_image, guide_image)
            #print('output: ',output)
            #input()
            #j+=1
            #continue
            ann_boxes = ann[0]['boxes']
            labels = ann[0]['labels']

            boxes = output[0]['boxes']
            ids = output[0]['labels']
            ids = np.asarray(ids.to('cpu'))
            boxes = np.asarray(boxes.to('cpu'))
            img = ref_image[0].to('cpu')
            img = img.permute(1,2,0).numpy()
            
            img_tempt = copy.deepcopy(img)

            image = make_boxes(img_tempt, boxes, ids)
            image = Image.fromarray(np.uint8(image*255))
            image.save('results/{}.jpg'.format(j))

            ann_image = make_boxes(img, ann_boxes, labels)
            ann_image = Image.fromarray(np.uint8(ann_image*255))
            ann_image.save('results/{}_ann.jpg'.format(j))

            j+=1
        print("That's it!")
