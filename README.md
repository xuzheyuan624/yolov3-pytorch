# yolov3-pytorch
A pytorch implementation of yolov3 <br>
This code is based on the official code of [YOLOv3](https://github.com/pjreddie/darknet), as well as a pytorch implementation 
[YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch).One of the goals of this code is training to get the result close
to the official one.So I improve the calculation of **loss functions** in ```yolo_loss.py``` and add lots of data augmentation such as **random cropping, 
flip horizontal, and multi-scale training** in ```data_transform.py```.I think I achieve a good result when I did this improvement.
## Requirements
------------------------------------------------------------------------------------------------------------------------------
1.Python 3.6<br>
2.PyTorch 0.4.1<br>
3.OpenCV<br>
4.imgaug<br>

## Installation
--------------
#### Get code
```
git clone https://github.com/xuzheyuan624/yolov3-pytorch.git
cd yolov3-pytorch
```
#### Download COCO dataset and COCO APi
```
cd data
bash get_coco_dataset.sh
```
#### Download Pascal VOC dataset
```
cd data
bash get_voc_dataset.sh
python voc_label.py
```
## Training
-------------------------
1.Download the pretrained darknet weights from [Google Drive](https://drive.google.com/file/d/1zoGUFk9Tfoll0vCteSJRGa9I1yEAfU-a/view?usp=sharing) or [Baidu Yun Drive](https://pan.baidu.com/s/18gwGWI11xMXlZvqvUPEhBQ) and move it to ```weights```<br>
2.Set up parameters in ```config.py```or```main.py``` <br>
#### Start training
```
python main.py train --name=coco
```
For training, each epoch trains on 117264 images from the train and validate COCO sets, and test on 4954 images from COCO dataset (**some images are lost :(**) 
#### Image Augmentation
```data_transfrom```shows many image augmentation, but the most effective augmentation is random cropping and multi-scale training, here is some examples for random cropping.<br>
![example 1](https://github.com/xuzheyuan624/yolov3-pytorch/blob/master/demo/step0_0.jpg)
![example 2](https://github.com/xuzheyuan624/yolov3-pytorch/blob/master/demo/step0_1.jpg)<br>
And for multi-scale training:

global step | image size |
----------- | ---------- |
  0 - 4000  |   13 * 32  |
4000 - 8000 |  ((0, 3) + 13) * 32 |
8000 - 12000 | ((0, 5) + 12) * 32 |
12000 - 16000 | ((0, 7) + 11) * 32 |
16000 -     | ((0, 9) + 10) * 32 |

## Evaluate
You can download the official weights from [Google Drive]() or [Baidu Yun Drive](https://pan.baidu.com/s/1Cr29v8L8i54sRjN6Cj3bqg) and put it in ```weights```<br>
Also you can use the weights trained by yourself just by modifying the path of weights in ```main.py``` or ```config.py```
<br>
#### Start evaluating
Use COCO APi to calculate mAP
```
python main.py eval --name==coco
```

