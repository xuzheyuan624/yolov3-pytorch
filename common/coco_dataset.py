import os
import numpy as np
import logging
import cv2
import random
import torch
from torch.utils.data import Dataset

from . import data_transforms


class COCODataset(Dataset):
    def __init__(self, list_path, img_size, is_training, batch_size, jitter, shuffle=False, seed=0, random=False, num_workers=1):
        self.batch_size = batch_size
        self.seed = seed
        self.random = random
        self.jitter = jitter
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.is_training = is_training
        self.img_files = []
        self.label_files = []
        for path in open(list_path, 'r'):
            label_path = path.replace('images', 'labels').replace('.png', '.txt').replace(
                '.jpg', '.txt').strip()
            if os.path.isfile(label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)
            else:
                logging.info("no label found. skip it: {}".format(path))
        logging.info("Total images: {}".format(len(self.img_files)))
        self.img_size = img_size  # (w, h)
        self.max_objects = 50
        self.max_num = 10000

        #  transforms and augmentation
        self.transforms = data_transforms.Compose()
        if is_training:
            self.transforms.add(data_transforms.ImageBaseAug())
            self.transforms.add(data_transforms.Crop(self.jitter))
            self.transforms.add(data_transforms.Flip(self.max_num))
        # self.transforms.add(data_transforms.KeepAspect())
        self.resize = data_transforms.ResizeImage()
        self.toTensor = data_transforms.ToTensor(self.max_objects)

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            logging.info("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)

        sample = {'image': img, 'label': labels}

        if self.random:
            if self.is_training and index % self.batch_size == 0:
                if self.seed < 4000:
                    self.img_size = 13 * 32
                elif self.seed < 8000:
                    self.img_size = (random.randint(0, 3) + 13) * 32
                elif self.seed < 12000:
                    self.img_size = (random.randint(0, 5) + 12) * 32
                elif self.seed < 16000:
                    self.img_size = (random.randint(0, 7) + 11) * 32
                else:
                    self.img_size = (random.randint(0, 9) + 10) * 32
                self.update()
        if self.transforms is not None and self.is_training:
            sample = self.transforms(sample)
        
        sample = self.resize(sample, self.img_size)
        sample = self.toTensor(sample)
        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample

    def __len__(self):
        return len(self.img_files)

    def random_shuffle(self):
        if self.shuffle:
            c = list(zip(self.img_files, self.label_files))
            random.shuffle(c)
            self.img_files, self.label_files = zip(*c)
    
    def update(self, global_step=None):
        if global_step is None:
            self.seed += self.num_workers
        else:
            self.seed = global_step


#  use for test dataloader
if __name__ == "__main__":
    dataloader = torch.utils.data.DataLoader(COCODataset("../data/coco/trainvalno5k.txt",
                                                         (416, 416), True, is_debug=True),
                                             batch_size=2,
                                             shuffle=False, num_workers=1, pin_memory=False)
    for step, sample in enumerate(dataloader):
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            image = np.transpose(image, (1, 2, 0))
            image *= 255
            image = image.astype(np.uint8)
            image = image.copy()
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("../demo/step{}_{}.jpg".format(step, i), image)
        # only one batch
        break
