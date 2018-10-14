# yolov3-pytorch
A pytorch implementation of yolov3 
This code is based on the official code of [YOLOv3](https://github.com/pjreddie/darknet), as well as a pytorch implementation 
[YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch).One of the goals of this code is training to get the result close
to the official one.So I improve the calculation of loss functions and add lots of data augmentation such as random cropping, 
flip horizontal, and multi-scale training.I think I achieve a good result when I did this improvement.
## Requirements
------------------------------------------------------------------------------------------------------------------------------
1.Python 3.6<br>
2.PyTorch 0.4.1
3.OpenCV
4.imgaug
