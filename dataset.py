import os
import torch
import random
from PIL import Image
from bs4 import BeautifulSoup
import torchvision.transforms as T


class GoogleOpenImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, obj_class_labels, max_images_per_class, train=False):
        self.root = root
        self.obj_class_labels = obj_class_labels
        self.train = train

        # load images and annotation
        imgs = []
        xml_files = []
        for obj_class in obj_class_labels:
            class_imgs = os.listdir(os.path.join(root, 'classes', obj_class, 'images'))
            class_img_filepaths = [
                os.path.join(root, f'classes/{obj_class}/images/{i}') for i in class_imgs
            ]

            class_xml_files = os.listdir(os.path.join(root, 'classes', obj_class, 'xml_files'))
            class_xml_filepaths = [
                os.path.join(root, f'classes/{obj_class}/xml_files/{i}') for i in class_xml_files
            ]

            imgs.extend(class_img_filepaths[:max_images_per_class])
            xml_files.extend(class_xml_filepaths[:max_images_per_class])

        self.imgs = list(sorted(imgs))
        self.xml_files = list(sorted(xml_files))

    @staticmethod
    def parse_xml(filename):
        xml_contents = open(filename).read()
        soup = BeautifulSoup(xml_contents, 'lxml')

        boxes = []
        labels = []
        for obj in soup.find_all('object'):
            xmin = int(obj.xmin.text)
            xmax = int(obj.xmax.text)
            ymin = int(obj.ymin.text)
            ymax = int(obj.ymax.text)

            # add check to make sure box is valid
            if (xmin >= xmax) or (ymin >= ymax):
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj.contents[1].text)

        return boxes, labels

    @staticmethod
    def flip(image, boxes):
        """
        Taken from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py

        Flip image horizontally.
        :param image: image, a PIL Image
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :return: flipped image, updated bounding box coordinates
        """
        # Flip image
        new_image = T.functional.hflip(image)

        # Flip boxes
        new_boxes = boxes
        new_boxes[:, 0] = image.width - boxes[:, 0] - 1
        new_boxes[:, 2] = image.width - boxes[:, 2] - 1
        new_boxes = new_boxes[:, [2, 1, 0, 3]]

        return new_image, new_boxes

    def __getitem__(self, idx):
        # load images and annotations
        img_path = self.imgs[idx]
        xml_path = self.xml_files[idx]

        img = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_xml(xml_path)

        # prepare target data
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        label_idxs = [self.obj_class_labels.index(label) + 1 for label in labels]

        # transformations
        img_transform_list = [T.ToTensor()]
        if self.train:
            if random.uniform(0, 1) > 0.5:
                img, boxes = self.flip(img, boxes)

        img_transforms = T.Compose(img_transform_list)
        img = img_transforms(img)

        return img, {
            'image_id': torch.tensor([idx]),
            'boxes': boxes,
            'labels': torch.as_tensor(label_idxs, dtype=torch.int64),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        }

    def __len__(self):
        return len(self.imgs)
