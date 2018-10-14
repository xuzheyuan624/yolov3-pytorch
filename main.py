import os
import sys
import numpy as np
import cv2
import time
import logging
import torch
from config import Config
from mode import Mode
from common.coco_dataset import COCODataset



class TrainConfig(Config):
    batch_per_gpu = 2
    parallels = [0]
    def __init__(self, name):
        super(TrainConfig, self).__init__()
        if name == 'coco':
            self.num_classes = 80
            self.train_list = './data/coco/trainvalno5k.txt'
            self.val_list = './data/coco/5k.txt'
        if name == 'voc':
            self.num_classes = 20
            self.train_list = './data/train.txt'
            self.val_list = './data/2007_test.txt'
    #pretrained_weights = './weights/model_backup.pth'
    #official_weights = './weights/official_yolov3_weights_pytorch.pth'


class EvalConfig(Config):
    batch_per_gpu = 8
    parallels = [0]
    def __init__(self, name):
        super(EvalConfig, self).__init__()
        if name == 'coco':
            self.val_list = './data/coco/5k.txt'
            #self.pretrained_weights = './weights/model_12.pth'
            self.official_weights = './weights/official_yolov3_weights_pytorch.pth'
            self.annotation = './data/coco/annotations/instances_val2014.json'
            self.image_size = 608
        elif name == 'voc':
            self.val_list = './data/2007_test.txt'
            self.pretrained_weights = ''

class InferenceConfig(Config):
    batch_per_gpu = 1
    parallels= [0]
    official_weights = './weights/official_yolov3_weights_pytorch.pth'




def train(config):

    model = Mode(config, True)

    train_dataloader = torch.utils.data.DataLoader(COCODataset(config.train_list, config.image_size, True, config.batch_size, config.jitter, shuffle=True, seed=config.seed, random=True, num_workers=config.num_workers),
                                                   batch_size = config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(COCODataset(config.val_list, config.image_size, False, config.batch_size, config.jitter),
                                                   batch_size = config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    model.train(train_dataloader, val_dataloader)
    

def evaluate(config, name):
    model = Mode(config, False)

    val_dataloader = torch.utils.data.DataLoader(COCODataset(config.val_list, config.image_size, False, config.batch_size, config.jitter, shuffle=False, seed=0, random=False, num_workers=config.num_workers),
                                                   batch_size = config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    if name == 'coco':
        model.eval_coco(val_dataloader)
    elif name == 'voc':
        model.eval_voc(val_dataloader)



def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255*np.random.rand(3)) for _ in range(N)]
    return colors

def test(config, name, path):

    if name == 'coco':
        classes = open('./data/coco.names', 'r').read().split('\n')[:-1]
        colors = random_colors(80)
    elif name == 'voc':
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        colors = random_colors(20)
    else:
        raise Exception('name must be coco or voc')

    if path is not None:
        logging.info('Read image from {}'.format(path))
        image = cv2.imread(path)
        if image is None:
            raise Exception('Read image error: {}'.format(path))
    else:
        raise Exception('You must input right image path')
    
    model = Mode(config, False)

    image, time = model.inference(image, classes, colors)
    logging.info('cost %.2f'%time)
    cv2.imwrite('prediction.jpg', image)

def demo(config, name, path=None):
    if name == 'coco':
        classes = open('./data/coco.names', 'r').read().split('\n')[:-1]
        colors = random_colors(80)
    elif name == 'voc':
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        colors = random_colors(20)
    else:
        raise Exception('name must be coco or voc')

    model = Mode(config, False)

    if path is not None:
        logging.info('Reading vedio from {}'.format(path))
        cap = cv2.VideoCapture(path)
    else:
        logging.info('Realtime detectiong')
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        image, time = model.inference(frame, classes, colors)
        logging.info('FPS : %.2f'%(float(1 / time)))
        cv2.imshow('frame', image)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', default=None, help='train, eval, test, or demo')
    parser.add_argument('--image', default=None, help='path to image')
    parser.add_argument('--video', default=None, help='path to vedio')
    parser.add_argument('--name', default='coco', help='coco or voc')
    args = parser.parse_args()
    
    if args.command == 'train':
        config = TrainConfig(args.name)
        config.display()
        # Start training
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.parallels))
        train(config)

    if args.command == 'eval':
        config = EvalConfig(args.name)
        config.display()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.parallels))
        evaluate(config, args.name)

    
    if args.command == 'test':
        config = InferenceConfig()
        config.display()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.parallels))
        test(config, args.name, args.image)

    if args.command == 'demo':
        config = InferenceConfig()
        config.display()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.parallels))
        demo(config, args.name, args.video)


if __name__ == "__main__":
    main()
