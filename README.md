# yolov3-pytorch
A pytorch implementation of yolov3 
This code is based on the official code of [YOLOv3](https://github.com/pjreddie/darknet), as well as a pytorch implementation 
[YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch).One of the goals of this code is training to get the result close
to the official one.So I improve the calculation of loss functions and add lots of data augmentation such as random cropping, 
flip horizontal, and multi-scale training.I think I achieve a good result when I did this improvement.
## Requirements
------------------------------------------------------------------------------------------------------------------------------
1.Python 3.6<br>
2.PyTorch 0.4.1<br>
3.OpenCV<br>
4.imgaug<br>

## Installation
--------------
### Get code
```
git clone https://github.com/xuzheyuan624/yolov3-pytorch.git
cd yolov3-pytorch
```
### Download COCO dataset and COCO APi
```
cd data
bash get_coco_dataset.sh
```
### Download Pascal VOC dataset
```
cd data
bash get_voc_dataset.sh
python voc_label.py
```
