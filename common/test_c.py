import torch
import time
import numpy
from _ext import bbox
from utils import bbox_iou

bbox1 = torch.FloatTensor([[3, 2.5, 2, 3]])
bbox2 = torch.FloatTensor([[5.5, 4.5, 5, 3], [1.5, 1, 3, 2]])
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s1 = time.time()
print(bbox_iou(bbox1, bbox2, center=True))
print('python:',time.time() - s1)
s2 = time.time()
ious = torch.zeros(bbox2.size(0), requires_grad=False)
bbox.bboxiou(bbox1, bbox2, ious, True)
print(ious)
print('c:', time.time() - s2)