import os
import sys
import numpy as np
import time
import cv2
import logging
import math
import xml.etree.ElementTree as ET
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from config import Config
from nets.model import Model
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def memory_usage_psutil():
    import psutil
    process = psutil.Process(os.getpid())
    men = process.memory_info()[0] / float(2 ** 20)
    print("men: {}".format(men))


class Mode():
    def __init__(self, config, is_training):
        self.config = config
        self.is_training = is_training
        self.net = Model(self.config, is_training=self.is_training)
        self.net.train(is_training)
        self.net.init_weights()
        if self.is_training:
            self.optimizer = self._get_optimizer()

        if len(self.config.parallels) > 0:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_loss = []
        for i in range(3):
            self.yolo_loss.append(YOLOLoss(config.anchors[i], config.image_size, config.num_classes))
        #if is_refine:
        #    self.refine_loss = RefineLoss(config.anchors, config.num_classes, (config.image_size, config.image_size))

        if config.pretrained_weights:
            logging.info("Load pretrained weights from {}".format(config.pretrained_weights))
            checkpoint = torch.load(config.pretrained_weights)
            state_dict = checkpoint['state_dict']
            self.net.load_state_dict(state_dict)
            self.epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint['global step'] + 1
        else:
            self.epoch = 0
            self.global_step = 0

        if config.official_weights:
            logging.info("Loading official weights from {}".format(config.official_weights))
            self.net.load_state_dict(torch.load(config.official_weights))
            self.global_step = 20000

    def _get_optimizer(self):
            optimizer = None

            # Assign different lr for each layer
            params = None
            base_params = list(
                map(id, self.net.backbone.parameters())
            )
            logits_params = filter(lambda p: id(p) not in base_params, self.net.parameters())

            if not self.config.freeze_backbone:
                params = [
                    {"params": self.net.parameters(), "lr": self.config.learning_rate},
                ]
            else:
                logging.info("freeze backbone's parameters.")
                for p in self.net.backbone.parameters():
                    p.requires_grad = False
                params = [
                    {"params": logits_params, "lr": self.config.learning_rate},
                ]

            # Initialize optimizer class
            if self.config.optimizer == "adam":
                optimizer = optim.Adam(params, weight_decay=self.config.weight_decay)
            elif self.config.optimizer == "amsgrad":
                optimizer = optim.Adam(params, weight_decay=self.config.weight_decay,
                                    amsgrad=True)
            elif self.config.optimizer == "rmsprop":
                optimizer = optim.RMSprop(params, weight_decay=self.config.weight_decay)
            else:
                # Default to sgd
                logging.info("Using SGD optimizer.")
                optimizer = optim.SGD(params, momentum=self.config.momentum,
                                    weight_decay=self.config.weight_decay,
                                    nesterov=(self.config.optimizer == "nesterov"))

            return optimizer
        
    
    def train(self, train_dataloader, val_dataloader):

        # Optimizer
        def adjust_learning_rate(optimizer, config, global_step):
            lr = config.learning_rate
            if global_step < config.burn_in:
                lr = lr * (global_step / config.burn_in) * (global_step / config.burn_in)
            elif global_step < config.decay_step[0]:
                lr = lr
            elif global_step < config.decay_step[1]:
                lr = config.decay_gamma * lr
            else:
                lr = config.decay_gamma * config.decay_gamma * lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            return lr
                
        summary = SummaryWriter(self.config.write)

        logging.info("Start training")
        while self.global_step < self.config.max_iter:
            #train step
            train_dataloader.dataset.random_shuffle()
            train_dataloader.dataset.update(self.global_step)
            for step, samples in enumerate(train_dataloader):
                images, labels = samples['image'], samples['label']
                images = images.cuda()
                image_size = images.size(2)
                batch_size = images.size(0)
                start_time = time.time()
                lr = adjust_learning_rate(self.optimizer, self.config, self.global_step)
                self.optimizer.zero_grad()
                outputs = self.net(images)
                losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls", "a"]
                losses = []
                for _ in range(len(losses_name)):
                    losses.append([])
                for i in range(3):
                    _loss_item = self.yolo_loss[i](outputs[i], labels, self.global_step)
                    for j, l in enumerate(_loss_item):
                        losses[j].append(l)
                losses = [sum(l) for l in losses]
                loss = losses[0]
                loss.backward()
                self.optimizer.step()
                #memory_usage_psutil()
                if step >= 0 and step % 10 == 0:
                    _loss = loss.item()
                    time_per_example = float(time.time() - start_time) / batch_size
                    logging.info(
                        "epoch [%.3d] step = %d size = %d loss = %.2f time/example = %.3f lr = %.5f loss_x = %.3f loss_y = %.3f loss_w = %.3f loss_h = %.3f loss_conf = %.3f loss_cls = %.3f loss_a = %.3f"%
                        (self.epoch, step, image_size, _loss, time_per_example, lr, losses[1], losses[2], losses[3], losses[4], losses[5], losses[6], losses[7])
                    )

                    summary.add_scalar("lr", lr, self.global_step)
                    for i, name in enumerate(losses_name):
                        v = _loss if i == 0 else losses[i]
                        summary.add_scalar(name, v, self.global_step)
                    
                    if step > 0 and step % 1000 == 0:
                        checkpoint_path = os.path.join(self.config.save_dir, "model_backup.pth")
                        checkpoint = {'state_dict':self.net.state_dict(), 'epoch': self.epoch, "global step": self.global_step}
                        torch.save(checkpoint, checkpoint_path)
                        logging.info("Model checkpoint saved to {}".format(checkpoint_path))
                
                self.global_step += 1

            checkpoint_path = os.path.join(self.config.save_dir, "model_{}.pth".format(self.epoch))
            checkpoint = {'state_dict':self.net.state_dict(), 'epoch': self.epoch, "global step": self.global_step}
            torch.save(checkpoint, checkpoint_path)
            logging.info("Model checkpoint saved to {}".format(checkpoint_path))


            #val every epoch
            logging.info('Start validating after epoch {}'.format(self.epoch))
            val_losses = []
            val_num = len(val_dataloader)
            for step, samples in enumerate(val_dataloader):
                images, labels = samples['image'], samples['label']
                with torch.no_grad():
                    outputs = self.net(images)
                    losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls", "a"]
                    losses = []
                    for _ in range(len(losses_name)):
                        losses.append([])
                    for i in range(3):
                        _loss_item = self.yolo_loss[i](outputs[i], labels)
                        for j, l in enumerate(_loss_item):
                            losses[j].append(l)
                    losses = [sum(l) for l in losses]
                    val_loss = losses[0].item()
                    if step > 0 and step % 10 == 0:
                        logging.info("Having validated [%.3d/%.3d]"%(step, val_num))
                    val_losses.append(val_loss)
            val_loss = np.mean(np.asarray(val_losses))
            logging.info("val loss = %.2f at epoch [%.3d]"%(val_loss, self.epoch))
            self.epoch += 1

    #def inference(self, inputs):
    #    with torch.no_grad():
    #        outputs = self.net(inputs)
    #        output = self.yolo_loss(outputs)
    #        detections = 




    def eval_coco(self, val_dataset):
        index2category = json.load(open("coco_index2category.json"))
        logging.info('Start Evaling')
        coco_result = []
        coco_img_ids = set([])

        for step, samples in enumerate(val_dataset):
            images, labels = samples['image'], samples['label']
            image_size = images.size(2)
            image_paths, origin_sizes = samples['image_path'], samples['origin_size']
            with torch.no_grad():
                outputs = self.net(images)
                #output = self.yolo_loss(outputs)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_loss[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, self.config.num_classes, conf_thres=0.01, nms_thres=0.4)
            for idx, detections in enumerate(batch_detections):
                image_id = int(os.path.basename(image_paths[idx])[-16:-4])
                coco_img_ids.add(image_id)
                if detections is not None:
                    origin_size = eval(origin_sizes[idx])
                    detections = detections.cpu().numpy()
                    dim_diff = np.abs(origin_size[0] - origin_size[1])
                    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
                    pad = ((pad1, pad2), (0, 0)) if origin_size[1] <= origin_size[0] else ((0, 0), (pad1, pad2))
                    scale = origin_size[0] if origin_size[1] <= origin_size[0] else origin_size[1]
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        x1 = x1 / self.config.image_size * scale
                        x2 = x2 / self.config.image_size * scale
                        y1 = y1 / self.config.image_size * scale
                        y2 = y2 / self.config.image_size * scale
                        x1 -= pad[1][0]
                        y1 -= pad[0][0]
                        x2 -= pad[1][0]
                        y2 -= pad[0][0]
                        w = x2 - x1
                        h = y2 - y1
                        coco_result.append({
                            "image_id":image_id,
                            "category_id":index2category[str(int(cls_pred.item()))],
                            "bbox":(float(x1), float(y1), float(w), float(h)),
                            "score":float(conf),
                        })
            logging.info("Now have finished [%.3d/%.3d]"%(step, len(val_dataset)))
        save_path = "coco_results.json"
        with open(save_path, "w") as f:
            json.dump(coco_result, f, sort_keys=True, indent=4, separators=(',', ':'))
        logging.info('Save result in {}'.format(save_path))
        
        logging.info('Using COCO APi to evaluate')
        cocoGt = COCO(self.config.annotation)
        cocoDt = cocoGt.loadRes(save_path)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.imgIds = list(coco_img_ids)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def eval_voc(self, val_dataset):
        logging.info('Start Evaling')
        
        def voc_ap(rec, prec, use_07_metric=False):
            """ ap = voc_ap(rec, prec, [use_07_metric])
            Compute VOC AP given precision and recall.
            If use_07_metric is true, uses the
            VOC 07 11 point method (default:False).
            """
            _rec = np.arange(0., 1.1, 0.1)
            _prec = []
            if use_07_metric:
                # 11 point metric
                ap = 0.
                for t in np.arange(0., 1.1, 0.1):
                    if np.sum(rec >= t) == 0:
                        p = 0
                    else:
                        p = np.max(prec[rec >= t])
                    _prec.append(p)
                    ap = ap + p / 11.
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mrec = np.concatenate(([0.], rec, [1.]))
                mpre = np.concatenate(([0.], prec, [0.]))

                # compute the precision envelope
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

            return ap
        
        def parse_rec(imagename):
            filename = imagename.replace('jpg', 'xml')
            tree = ET.parse(filename)
            objects = []
            for obj in tree.findall('object'):
                obj_struct = {}
                obj_struct['pose'] = obj.fine('pose').text()
                obj_struct['truncated'] = int(obj.find('truncated').text)
                obj_struct['difficult'] = int(obj.find('difficult').text)
                bbox = obj.find('bndbox')
                obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                      int(bbox.find('ymin').text),
                                      int(bbox.find('xmax').text),
                                      int(bbox.find('ymax').text)]
                objects.append(obj_struct)
            return objects


        for step, samples in enumerate(val_dataset):
            images, labels = samples['image'], samples['label']
            image_paths, origin_sizes = samples['image_path'], samples['origin_size']
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_loss[i](outputs[i], labels))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, self.config.num_classes, conf_thres=0.01, nms_thres=0.4)
            for idx, detections in enumerate(batch_detections):
                image_id = int(os.path.basename(image_paths[idx])[:6])
                if detections is not None:
                    origin_size = eval(origin_sizes[idx])
                    detections = detections.cpu().numpy()
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        x1 = x1 / self.config.image_size * origin_size[0]
                        x2 = x2 / self.config.image_size * origin_size[0]
                        y1 = y1 / self.config.image_size * origin_size[1]
                        y2 = y2 / self.config.image_size * origin_size[1]




    def inference(self, image, classes, colors):

        image_origin = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config.image_size, self.config.image_size), interpolation=cv2.INTER_LINEAR)
        image = np.expand_dims(image, 0)
        image = image.astype(np.float32)
        image /= 255
        image = np.transpose(image, (0, 3, 1, 2))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        
        start_time = time.time()
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            outputs = self.net(image)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_loss[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, self.config.num_classes, conf_thres=0.5, nms_thres=0.4)
            spand_time = float(time.time() - start_time)
        detection = batch_detections[0]
        if detection is not None:
            origin_size = image_origin.shape[:2]
            detection = detection.cpu().numpy()
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                x1 = int(x1 / self.config.image_size * origin_size[1])
                x2 = int(x2 / self.config.image_size * origin_size[1])
                y1 = int(y1 / self.config.image_size * origin_size[0])
                y2 = int(y2 / self.config.image_size * origin_size[0])
                color = colors[int(cls_pred)]
                image_origin = cv2.rectangle(image_origin, (x1,y1), (x2,y2), color, 3)
                image_origin = cv2.rectangle(image_origin, (x1,y1), (x2,y1+20), color, thickness=-1)
                caption = "{}:{:.2f}".format(classes[int(cls_pred)], cls_conf)
                image_origin = cv2.putText(
                    image_origin, caption, (x1,y1+15), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2
                )
            return image_origin, spand_time

    
