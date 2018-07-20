
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *
import json
class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=32, num_workers=4):
        with open(root, 'r') as file:
            self.lines = json.load(file)

       
        if train == True:
            self.nSamples = 10000
        else:
            self.nSamples = len(self.lines[0])
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
       
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        if self.train and index % 32== 0:
            if self.seen < 4000*32:
               #width = 13*32
               #height = 13*32
               width = 22*8
               height = 40*8
               self.shape = (height, width)
            elif self.seen < 8000*32:
               width = (random.randint(-3,3) + 22)*8
               height = (random.randint(-3,3) + 40)*8
               self.shape = (height, width)
            elif self.seen < 12000*32:
               width = (random.randint(-5,5) + 22)*8
               height = (random.randint(-5,5) + 40)*8
               self.shape = (height, width)
            elif self.seen < 16000*32:
               width = (random.randint(-7,7) + 22)*8
               height = (random.randint(-7,7) + 40)*8
               self.shape = (height, width)
            else: # self.seen < 20000*64:
               width = (random.randint(-9,9) + 22)*8
               height = (random.randint(-9,9) + 40)*8
               self.shape = (height, width)

        if self.train:
            cls = random.choice(self.lines)
            info = random.choice(cls)
            imgpath = info[0]
            box = info[1]
            
            label = np.zeros(10)
            label[1:3] = box[2:4]
            label[3:5] = box[0:2]
            
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath,label, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
            
            
        else:
            info = self.lines[0][index]
            imgpath = info[0]
            box = info[1]
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            label = np.zeros(10)
            label[1:3] = box[2:4]
            label[3:5] = box[0:2]
            label = torch.from_numpy(label)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.batch_size
        return (img, label)