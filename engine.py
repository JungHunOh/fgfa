import math
import sys
import time
import torch
import os
import torchvision.models.detection.mask_rcnn
from PIL import Image
import glob
import torchvision.transforms as transforms 
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils

def guide_images_processing(guide_images, device):
    '''
    return: list[dict]

    'image': image tensor [C, H, W]
    'bef', 'aft': list[int] -> list of the id of reference image.
                                ex: [302,304] means this image is guide image of id 302, 304 reference image.
    'ref': int -> if this image is one of the reference images, 'ref' is its id.
    '''
    results = []
    images = []
    ref_ids = []

    for guide_img in guide_images:
        images += guide_img['bef']
        images += guide_img['aft']
        ref_ids.append(guide_img['ref_id']) 

    imgs = []

    for image in images:
        if len(imgs) == 0:
            imgs.append(image)
        else:
            test = 0
            for img in imgs:
                if torch.equal(image, img):
                    test =1
            if test == 0:
                imgs.append(image)
    
    images = imgs
    
    for image in images:

        result = dict()
        result['bef'] = []
        result['aft'] = []
        result['ref'] = -1
        for guide_img in guide_images:

            for i, bef_img in enumerate(guide_img['bef']):
                if torch.equal(image, bef_img):
                    result['bef'].append(guide_img['ref_id'])
                    image_id = guide_img['bef_id'][i]

                    if image_id in ref_ids:
                        result['ref'] = image_id

            for i, aft_img in enumerate(guide_img['aft']):
                if torch.equal(image, aft_img):
                    result['aft'].append(guide_img['ref_id'])
                    image_id = guide_img['aft_id'][i]

                    if image_id in ref_ids:
                        result['ref'] = image_id

        result['image'] = image.to(device)

        results.append(result)
                
    
    return results


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, lr_scheduler, summary):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    #lr_scheduler = None
    #if epoch == 0:
    #    warmup_factor = 1. / 1000
    #    warmup_iters = min(1000, len(data_loader) - 1)

    #    lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    for iteration, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        

        ref_images, guide_images, targets = data
        ref_images = list(image.to(device) for image in ref_images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        guide_images = guide_images[0]

        loss_dict = model(ref_images, guide_images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if iteration != 0 and iteration % 10 == 0:
            summary.add_scalar('loss', loss_value, iteration + 4464 * epoch)
            summary.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], iteration + 4464 * epoch)

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    
    print_freq = 50

    for ref_images, guide_images, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        ref_images = list(image.to(device) for image in ref_images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        guide_images = guide_images_processing(guide_images, device)      

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(ref_images, guide_images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
