import os

class Config():
    #backbone params
    backbone_name = "darknet53"
    backbone_pretrained = './weights/darknet53_weights_pytorch.pth'

    #yolo params
    anchors = [[[116, 90], [156, 198], [373, 326]],
               [[30, 61], [62, 45], [59, 119]],
               [[10, 13], [16, 30], [33, 23]]]
    num_classes = 80

    #learning rate
    learning_rate = 0.001
    burn_in = 1000
    freeze_backbone = True
    decay_gamma = 0.1
    decay_step = [400000, 450000]
    optimizer = 'sgd'
    weight_decay = 4e-5
    momentum = 0.9
    use_focalloss=False

    #training params
    batch_per_gpu = 16
    num_workers = 2
    seed = 0
    train_list = ''
    val_list = ''
    max_iter = 500200
    image_size = 416
    jitter = 0.3
    parallels = [0]
    save_dir = 'weights'
    logs = 'logs'
    pretrained_weights = ''
    official_weights = ''


    def __init__(self):
        if len(self.parallels) > 0:
            self.batch_size = len(self.parallels) * self.batch_per_gpu
        else:
            self.batch_size = self.batch_per_gpu
        self.write = os.path.join(self.save_dir, self.logs)
    
    def display(self):
        print("\nConfiguration values")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
