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
import json
import time
import backbone
import random
import glob
from torchvision import datasets, transforms
from engine import train_one_epoch, evaluate
import utils
import argparse
from tensorboardX import SummaryWriter
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from engine import _get_iou_types, guide_images_processing

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='settings for eval')

    parser.add_argument('--num', type=int)

    args = parser.parse_args()

    print('args: ',args)

    # train on the gpu or on the cpu, if a gpu is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 7

    # load a pre-trained model for classification and return
    # only the features
    backbone = backbone.FGFA_Backbone() 
    # fasterrcnn needs to know the number of
    # output channels in a backbone. for mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1024

    # let's make the rpn generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. we have a tuple[tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_sizes = ((16, 32, 64, 128),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a tensor, featmap_names is expected to
    # be [0]. more generally, the backbone should return an
    # ordereddict[tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names='0',
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FGFAFasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler,
                   rpn_pre_nms_top_n_train=12000,
                   rpn_pre_nms_top_n_test=6000)


    # move model to the right device

    val_folder = '../drone/drone-dataset/t1_video/val'
    val_dataset = MyDataset(val_folder, Is_training = False)
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    n_threads = torch.get_num_threads()

    torch.set_num_threads(1)
    cpu_device = torch.device('cpu')
    
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)

    for i in range(0, 56):
        coco_evaluator = CocoEvaluator(coco, iou_types)
        
        model.to(device)

        print('load model {} parameters...'.format(i))
        model.load_state_dict(torch.load('output/output_epoch_{}.pt'.format(i)))
    
        model.eval()

        data_iter = iter(data_loader)

        for data in data_iter:
            images, guide_images, targets = data

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            guide_images = guide_images_processing(guide_images, device)

            torch.cuda.synchronize()
            outputs = model(images, guide_images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            res = {target["image_id"].item(): output for target, output in zip(targets,outputs)}

            coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()

        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)   
    
    print("That's it!")
