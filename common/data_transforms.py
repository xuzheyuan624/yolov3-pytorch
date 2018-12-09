import numpy as np
import cv2
import torch
import random
#  pip install imgaug
import imgaug as ia
from imgaug import augmenters as iaa

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def box2x1x2y1y2(boxes, h, w):
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]
    boxes[:, 0] = boxes[:, 0] * w
    boxes[:, 2] = boxes[:, 2] * w
    boxes[:, 1] = boxes[:, 1] * h
    boxes[:, 3] = boxes[:, 3] * h
    return boxes


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)


class ToTensor(object):
    def __init__(self, max_objects=50):
        self.max_objects = max_objects

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)

        filled_labels = np.zeros((self.max_objects, 5), np.float32)
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(filled_labels)}

class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        image_new = np.pad(image, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = image_new.shape

        # Extract coordinates for unpadded + unscaled image
        x1 = w * (label[:, 1] - label[:, 3]/2)
        y1 = h * (label[:, 2] - label[:, 4]/2)
        x2 = w * (label[:, 1] + label[:, 3]/2)
        y2 = h * (label[:, 2] + label[:, 4]/2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        label[:, 1] = ((x1 + x2) / 2) / padded_w
        label[:, 2] = ((y1 + y2) / 2) / padded_h
        label[:, 3] *= w / padded_w
        label[:, 4] *= h / padded_h

        return {'image': image_new, 'label': label}

class KeepAspectResize(object):
    def __init__(self, interpolation=cv2.INTER_LINEAR):
        self.interpolation = interpolation

    def __call__(self, sample, img_size):
        image, label = sample['image'], sample['label']
        h, w, _ = image.shape
        new_size = tuple((int(img_size), int(img_size * h / w))) if h < w else tuple((int(img_size * w / h), int(img_size)))
        image_resize = cv2.resize(image, new_size, interpolation=self.interpolation)
        dim_diff = np.abs(new_size[0] - new_size[1])
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if new_size[1] <= new_size[0] else ((0, 0), (pad1, pad2), (0, 0))
        image_new = np.pad(image_resize, pad, 'constant', constant_values=127.5)
        padded_h, padded_w, _ = image_new.shape

        x1 = new_size[0] * (label[:, 1] - label[:, 3]/2)
        x2 = new_size[0] * (label[:, 1] + label[:, 3]/2)
        y1 = new_size[1] * (label[:, 2] - label[:, 4]/2)
        y2 = new_size[1] * (label[:, 2] + label[:, 4]/2)

        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]

        label[:, 1] = ((x1 + x2) / 2) / padded_w
        label[:, 2] = ((y1 + y2) / 2) / padded_h
        label[:, 3] *= new_size[0] / padded_w
        label[:, 4] *= new_size[1] / padded_h

        return {'image': image_new, 'label': label}


class ResizeImage(object):
    def __init__(self, interpolation=cv2.INTER_LINEAR):
        self.interpolation = interpolation
    def __call__(self, sample, img_size):
        new_size = tuple((img_size, img_size))
        image, label = sample['image'], sample['label']
        #print("Size is : {}".format(self.new_size))
        image = cv2.resize(image, new_size, interpolation=self.interpolation)
        return {'image': image, 'label': label}

class Flip(object):
    def __init__(self, max_num):
        self.max_num = max_num

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        flip = random.randint(1, self.max_num)%2
        #print('flip: ',flip)
        if flip:
            #print('flipping')
            image = image[:, ::-1, :]
            label[:, 1] = 0.999 - label[:, 1]
        return {'image':image, 'label':label}



class Crop(object):
    def __init__(self, jitter):
        self.jitter = jitter
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None)
        )
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        origin_h, origin_w = image.shape[:2]
        boxes = label[:, 1:].copy()
        boxes = box2x1x2y1y2(boxes, origin_h, origin_w) #center boxes
        while True:
            #random.seed(get_random_seed())
            mode = random.choice(self.sample_options)
            if mode is None:
                return sample
            
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float(0)
            if max_iou is None:
                max_iou = float('inf')
            
            for _ in range(50):
                origin_image = image
                #random.seed(get_random_seed())
                w = random.uniform(self.jitter * origin_w, origin_w)
                #random.seed(get_random_seed())
                h = random.uniform(self.jitter * origin_h, origin_h)

                if h / w < 0.5 or h / w > 2.0:
                    continue

                #random.seed(get_random_seed())
                left = random.uniform(0, origin_w - w)
                #random.seed(get_random_seed())
                top = random.uniform(0, origin_h - h)

                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                overlap = jaccard_numpy(boxes, rect)

                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                
                current_image = origin_image[rect[1]:rect[3], rect[0]:rect[2], :]
                current_shape = current_image.shape
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                mask = m1 * m2

                if not mask.any():
                    continue
                
                #take only matching boxes
                current_boxes = boxes[mask, :].copy()
                current_label = label[:, 0][mask].copy()

                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                current_boxes[:, 2:] = current_boxes[:, 2:] - current_boxes[:, :2]
                current_boxes[:, :2] = current_boxes[:, :2] + current_boxes[:, 2:] / 2

                current_boxes[:, 0], current_boxes[:, 2] = current_boxes[:, 0] / current_shape[1], current_boxes[:, 2] / current_shape[1]
                current_boxes[:, 1], current_boxes[:, 3] = current_boxes[:, 1] / current_shape[0], current_boxes[:, 3] / current_shape[0]
                
                #print(mask)
                #print('choosing {}, rect: {}'.format(mode, rect))
                current_label = np.concatenate((np.expand_dims(current_label, 1), current_boxes), 1)
                #print('current label: ', current_label)
                return {'image':current_image, 'label':current_label}


class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                # Add gaussian noise to some images.
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # Add a value of -5 to 5 to each pixel.
                sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                # Change brightness of images (80-120% of original value).
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                # Improve or worsen the contrast of images.
                sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def __call__(self, sample):
        seq_det = self.seq.to_deterministic()
        image, label = sample['image'], sample['label']
        image = seq_det.augment_images([image])[0]
        return {'image': image, 'label': label}
