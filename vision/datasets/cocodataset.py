import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
import sys
from pycocotools.coco import COCO
import cv2
import numpy as np
import copy


training_annFile = "data/COCO/annotations/instances_train.json"
val_annFile = "data/COCO/annotations/instances_val.json"

class COCODataSet:
    def __init__(self, root,annotations_file, transform=None, target_transform=None):
        self.coco = COCO(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        self.ids_image = sorted(self.coco.getImgIds())
        # self.ids_ann = self.coco.getAnnIds()
        self.ids = self.ids_image
        self.root = root;
        self.class_names = (
            "BACKGROUND",
            'person',
            'bicycle',
            'car',
            'motorcycle',
            'airplane',
            'bus',
            'train',
            'truck',
            'boat',
            'traffic light',
            'fire hydrant',
            'stop sign',
            'parking meter',
            'bench',
            'bird',
            'cat',
            'dog',
            'horse',
            'sheep',
            'cow',
            'elephant',
            'bear',
            'zebra',
            'giraffe',
            'backpack',
            'umbrella',
            'handbag',
            'tie',
            'suitcase',
            'frisbee',
            'skis',
            'snowboard',
            'sports ball',
            'kite',
            'baseball bat',
            'baseball glove',
            'skateboard',
            'surfboard',
            'tennis racket',
            'bottle',
            'wine glass',
            'cup',
            'fork',
            'knife',
            'spoon',
            'bowl',
            'banana',
            'apple',
            'sandwich',
            'orange',
            'broccoli',
            'carrot',
            'hot dog',
            'pizza',
            'donut',
            'cake',
            'chair',
            'couch',
            'potted plant',
            'bed',
            'dining table',
            'toilet',
            'tv',
            'laptop',
            'mouse',
            'remote',
            'keyboard',
            'cell phone',
            'microwave',
            'oven',
            'toaster',
            'sink',
            'refrigerator',
            'book',
            'clock',
            'vase',
            'scissors',
            'teddy bear',
            'hair drier',
            'toothbrush'
        )
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        boxes = []
        labels = []
        # for x in range(len(self.ids_ann)):
        
        # anns = self.coco.loadAnns(self.ids_ann[image_id])[0]
        while (len(self.coco.getAnnIds(image_id)) == 0):
            image_id = self.ids[idx+1]
            
        anns = self.coco.loadAnns(self.coco.getAnnIds(image_id))[0]

        tempBox = anns["bbox"]
        

        x = tempBox[0]
        y = tempBox[1]
        width = tempBox[2]
        height = tempBox[3]
        tempBox[2] = width+x
        tempBox[3] = height+y
        boxes.append(tempBox)
        catId = anns["category_id"]
        labels.append(self.class_dict[self.coco.loadCats(catId)[0]["name"]])
                    
        #print('__getitem__  image_id=' + str(image_id) + ' \nboxes=' + str(boxes) + ' \nlabels=' + str(labels))
        image_file = self.coco.loadImgs(image_id)
        
        if image_file is None:
            raise IOError('failed to load ' + image_file)
        path = self.coco.loadImgs(image_id)[0]["file_name"]
        image = cv2.imread(str(os.path.join(self.root, path)))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = np.array(boxes,dtype="float32")
        labels = np.array(labels,dtype='int64')
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        
        # tempTarget = {}
        # tempTarget["boxes"] = boxes
        # tempTarget["labels"] = labels
        # target = [tempTarget]
        # return image, target
        imagePath = str(os.path.join(self.root, path))
        return image, boxes, labels