import os
import cv2
import random
import numpy as np

import torch


##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, img_size, transform=None):
        self.df = df
        self.img_size = img_size
        self.transform = transform
        self.arr_image = df['path'].values   # already split kfold, don't use unique()

    def __len__(self) -> int:    # -> is return function annotation
        return len(self.arr_image)

    def __getitem__(self, index):
        image_id = self.arr_image[index]

        if random.random() > 0.5:
            image, boxes = self.load_image_and_boxes(index)
        else:
            # cut mix output이 너무 커서 input data로 만들 수 없으면 에러 발생
            try:
                image, boxes = self.load_image_and_boxes(index)
            except:
                image, boxes = self.load_image_and_boxes(index)

        # label has only one class (background 0, object 1)
        # boxes.shape is (n_boxes, 4). So boxes.shape[0] would be n_boxes.
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)    # tensor([1, 1, ..., 1]) // num of 1 is n_boxes

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['img_scale'] = torch.tensor([1.])
        target['image_id'] = torch.tensor([index])  # It's not important >_<
        target['img_size'] = torch.tensor([(self.img_size, self.img_size)])

        for box in boxes:
            cv2.rectangle(boxes,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 255), 2)

            # ax.set_axis_off()
        cv2.imwrite('sample.jpg', image)

        if self.transform:
            # Because it is possible for transformation to remove all bboxes in a image,
            # have to use 'while' code.
            while True:
                # If image resize, boxes should resize also simultaneously.
                sample = {'image': image, 'bboxes': boxes, 'labels': labels}
                sample = self.transform(**sample)

                # if transform, some boxes would delete.
                # assert len(sample['bboxes']) == labels.shape[0], 'not equal!'

                # After Augmentation,
                # if image has no boxes, thus len(sample['bboxes'])=0 is True, loop again
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    # EfficientNet implementation in the effdet library takes bboxes in yxyx format
                    target['boxes'] = torch.tensor(sample['bboxes'])
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]
                    target['labels'] = torch.stack(sample['labels']) # have to add this
                    break

        return image, target, image_id


##
# we don't use, right now
    def refine_boxes(self, image, boxes):
        img_x, img_y = image.shape[1], image.shape[0]

        # an image coordinate starts from left-top (0,0)
        # and right +, down +
        cropped_image = image[img_y // 2:, :img_x]

        result_boxes = []
        for box in boxes:
            # if w,h too small, remove one.
            if box[2] < 5 or box[3] < 5:
                continue

            # 만약 box y result가 0보다 작은 경우, 0으로 리턴
            cropped_y = box[1] - img_y // 2
            if cropped_y < 0:
                box[1] = 0
            else:
                box[1] = cropped_y
            result_boxes.append(box)

        result_boxes = np.array(result_boxes)

        return cropped_image, result_boxes


##
    def load_image_and_boxes(self, index):
        # input image
        image = cv2.imread('../../data/' + self.arr_image[index], cv2.IMREAD_COLOR)
        # In open-cv, Image read is BGR order. and Albumentation use RGB format.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)


        # boxes
        # a bounding box formatted as a python-style (list of [xmin, ymin, width, height])
        path = self.arr_image[index]
        records = self.df[self.df['path'] == path ]
        boxes = records[['x', 'y', 'w', 'h']].values    # np array ([[..., ..., ..., ...], ...])

        # refine boxes
        # cropped_image, boxes = self.refine_boxes(image, boxes)

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]    # xmax
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]    # ymax






        return image, boxes


##
    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """

        w, h = imsize, imsize
        s = imsize // 2

        paths = self.df['path']
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, paths.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)


            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # if y2b-y1b or x2b-x1b is larger than image size, error comes out
            # img_size needs to be changed.
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]


        # if BBox in an image is too small, upper method returns empty list. thus, no detected region. ex> car plate
        # So we need to change this.
        if len(result_boxes) == 0 :
            result_image, result_boxes = self.load_image_and_boxes(index)

        return result_image, result_boxes