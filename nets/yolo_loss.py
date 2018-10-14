import torch
import torch.nn as nn
import numpy as np
import math

from common.utils import bbox_iou
from common._ext import bbox

class YOLOLoss(nn.Module):
    def __init__(self, anchors, img_size, num_classes):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 5.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)

    def forward(self, input, targets=None, global_step=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size / in_h
        stride_w = self.img_size / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # Calculate offsets for each grid
        grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
            bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
            bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        # Calculate anchor w, h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if targets is None:
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data
        
        else:
            n_obj, mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors,
                                                                           in_w, in_h, pred_boxes.cpu().detach(), 
                                                                           self.ignore_threshold)
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            #  losses.
            loss_x = self.mse_loss(x * mask, tx * mask) / (2 * n_obj)
            loss_y = self.mse_loss(y * mask, ty * mask) / (2 * n_obj)
            loss_w = self.mse_loss(w * mask, tw * mask) / (2 * n_obj)
            loss_h = self.mse_loss(h * mask, th * mask) / (2 * n_obj)
            loss_conf = self.bce_loss(conf * mask, mask) / n_obj + \
                0.2 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0) / n_obj
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1]) / n_obj
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
            
            if global_step is not None and global_step < 12800:
                axy = torch.zeros_like(x, requires_grad=False).cuda()
                awh = torch.zeros_like(w, requires_grad=False).cuda()
                axy.fill_(0.5)
                a_loss = (self.mse_loss(x, axy) + self.mse_loss(y, axy) + self.mse_loss(w, awh) + self.mse_loss(h, awh)) / (2 * n_obj)
                loss = loss + 0.1 * a_loss
                anchor_loss = a_loss.item()
            else:
                anchor_loss = 0

            return loss, loss_x.item(), loss_y.item(), loss_w.item(),\
                loss_h.item(), loss_conf.item(), loss_cls.item(), anchor_loss
    
    def get_target(self, target, anchors, in_w, in_h, pred_box, ignore_threshold):
        bs = target.size(0)
        n_obj = 0
        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(bs):
            box_p = pred_box[b].view(-1, 4)
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                n_obj += 1
                # Convert to position relative to box
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box_match = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box_match, anchor_shapes, True)
                pred_ious = bbox_iou(gt_box, box_p, True).view(self.num_anchors, in_h, in_w)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b][pred_ious >= ignore_threshold] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)
                best_conf = pred_ious[best_n, gj, gi]
                # Masks
                mask[b, best_n, gj, gi] = 1
                noobj_mask[b, best_n, gj, gi] = 0
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = torch.log(gw/anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = torch.log(gh/anchors[best_n][1] + 1e-16)
                # object
                tconf[b, best_n, gj, gi] = best_conf
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return n_obj, mask, noobj_mask, tx, ty, tw, th, tconf, tcls
