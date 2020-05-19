import os
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import json
import backbone
import random
import glob

train_video_idxes = [301, 601, 902, 1174, 1475, 1775, 2075, 2376, 2677, 2977,
            3277, 3577, 3878, 4178, 4478, 4779, 5080, 5381, 5681, 5982,
            6283, 6583, 6884, 7184, 7484, 7785, 8085, 8386, 8686, 8986,
            9192, 9493, 9793, 10094, 10395, 10696, 10996, 11290, 11590,
            11890, 12190, 12490, 12790, 13090, 13390]

val_video_idxes = [300, 600, 900, 1200, 1500]

class MyDataset(data.Dataset):
    def __init__(self, root, K=7, num_neighbor=5,transforms=transforms.ToTensor(), Is_training=True):
        self.root = root
        self.K = K
        self.num_neighbor = num_neighbor
        self.is_training = Is_training
        self.transforms = transforms
        self.frames = sorted(glob.glob(os.path.join(root,'t1_video_*/*.jpg')))
        self.num_frames = len(self.frames)
        if self.is_training is True:
            self.annotations = json.load(open('../drone/drone-dataset/annotations/annotations_train.json'))
            self.video_idxes = train_video_idxes
        else:
            self.annotations = json.load(open('../drone/drone-dataset/annotations/annotations_val.json'))
            self.video_idxes = val_video_idxes

    def __getitem__(self, idx):
        
        assert idx < self.num_frames  

        ref_idx = idx 
         
        ref_image = self.frames[ref_idx]
        ref_image = Image.open(ref_image)
        ref_image = self.transforms(ref_image)
        
        for i, video_idx in enumerate(self.video_idxes):
            if video_idx == self.video_idxes[0]:
                if idx <= video_idx:
                    min_idx = 0
                    max_idx = video_idx-1
            else:
                if idx <= video_idx and idx > self.video_idxes[i-1]:
                    min_idx = self.video_idxes[i-1]
                    max_idx = video_idx-1
        
        start_frame_idx = max(min_idx, ref_idx-self.K)
        last_frame_idx = min(max_idx, ref_idx+self.K)
        guide_images = []
        #guide_images['bef_id'] = []
        #guide_images['aft_id'] = []
        #guide_images['ref_id'] = ref_idx + 1
        
        image_bef_ids = [i for i in range(start_frame_idx + 1, ref_idx + 1)]
        image_aft_ids = [i for i in range(ref_idx + 1 + 1, last_frame_idx + 1 + 1)]

        if self.is_training is True:
            if len(image_bef_ids) < self.num_neighbor:
                sampled_bef_ids = image_bef_ids
            else:
                sampled_bef_ids = sorted(random.sample(image_bef_ids, self.num_neighbor))

            if len(image_aft_ids) < self.num_neighbor:
                sampled_aft_ids = image_aft_ids
            else:
                sampled_aft_ids = sorted(random.sample(image_aft_ids, self.num_neighbor))

            sampled_ids = sampled_bef_ids + sampled_aft_ids

        else:
            sampled_ids = image_bef_ids + image_aft_ids

        for i in sampled_ids:
            
            index = i-1

            if index < ref_idx:
                img = self.frames[index]
                img = Image.open(img)
                img = self.transforms(img)
                guide_images.append(img)
                #guide_images['bef_id'].append(i)

            elif index > ref_idx:
                img = self.frames[index]
                img = Image.open(img)
                img = self.transforms(img)

                guide_images.append(img)
                #guide_images['aft_id'].append(i)

        annotations = self.annotations["annotations"]
        
        anns={}
        anns['boxes']=[]
        anns['labels']=[]
        anns['area']=[]
        anns['iscrowd']=[]
        
        exist_box=False
        for annotation in annotations:
            im_id=annotation["image_id"]
            if idx  == im_id:
                exist_box = True
                xmin=annotation['bbox'][0]
                ymin=annotation['bbox'][1]
                xmax=annotation['bbox'][0]+annotation['bbox'][2]
                ymax=annotation['bbox'][1]+annotation['bbox'][3]
                anns['boxes'].append([xmin, ymin, xmax, ymax])
                anns['labels'].append(annotation['category_id'])
                anns['area'].append(annotation['area'])
                anns['iscrowd'].append(annotation['iscrowd'])
        
        if exist_box == False:
            anns["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            anns["labels"] = torch.zeros((1, 1), dtype=torch.int64)
            anns["image_id"] = torch.tensor([idx + 1])
            anns["area"] = torch.zeros((0,), dtype=torch.int64) 
            anns["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        else:
            anns['image_id']=torch.tensor([idx + 1])
            anns['boxes']=torch.as_tensor(anns['boxes'], dtype=torch.float32)
            anns['labels']=torch.as_tensor(anns['labels'], dtype=torch.int64)
            anns['iscrowd']=torch.as_tensor(anns['iscrowd'], dtype=torch.int64)
            anns['area']=torch.as_tensor(anns['area'], dtype=torch.float32)
        
        return ref_image, guide_images ,anns
    
    def __len__(self):
        return self.num_frames


