import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter, ImageDraw, ImageStat
import h5py
import cv2
import glob
import math

class CSRNetDataset(Dataset):
    def __init__(self, config, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        # if train:
        #     root = root *4
        # random.shuffle(root)
        
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lines = []

        root = str(root)

        if config.dataset == 'UCFCC50':
            buffer = []
            for i in range(1, 6):
                if i == config.cc50_val:
                    self.val_lines = glob.glob(os.path.join(root, 'fold_'+str(i), 'images', '*.jpg'))
                else:
                    buffer.append(glob.glob(os.path.join(root, 'fold_'+str(i), 'images', '*.jpg')))
            
            for item in buffer:
                for x in item:
                    self.lines.append(x)
            self.nSamples = len(self.lines)
        else:
            self.lines = glob.glob(os.path.join(root, 'train', 'images', '*.jpg'))
            self.val_lines = glob.glob(os.path.join(root, 'images', '*.jpg').replace('train', 'val'))
            self.nSamples = len(self.lines)
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]

        img, target = self.load_data(img_path)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883

        if self.transform is not None:
            img = self.transform(img)

        return img,target

    def get_lines(self):
        return self.lines

    def get_val_lines(self):
        return self.val_lines

    def load_data(self, img_path, train = True):
        gt_path = img_path.replace('.jpg','.h5').replace('images','density_maps')
        img = Image.open(img_path).convert('RGB')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        
        if train:
            crop_size = (img.size[0]/2,img.size[1]/2)
            if random.randint(0,9)<= -1: 
                dx = int(random.randint(0,1)*img.size[0]*1./2)
                dy = int(random.randint(0,1)*img.size[1]*1./2)
            else:
                dx = int(random.random()*img.size[0]*1./2)
                dy = int(random.random()*img.size[1]*1./2)

            img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
            target = target[dy:int(round(crop_size[1]+dy)),dx:int(round(crop_size[0]+dx))]
            # target = target[dy:int(crop_size[1])+dy,dx:int(crop_size[0])+dx]

            if random.random()>0.8:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
        target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
        
        return img,target