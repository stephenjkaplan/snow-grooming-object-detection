import os
import torch
import random
from PIL import Image
from bs4 import BeautifulSoup
import torchvision.transforms as T


class GoogleOpenImageDataset(torch.utils.data.Dataset):
    """
    Inherits from the PyTorch Dataset class to work specifically for Google Open Images data format.
    Prepares dataset for training or validation.
    """
    def __init__(self, obj_class_labels, max_images_per_class, folder_name='data', train=False):
        """
        Initializes class.

        :param list obj_class_labels: Object labels to detect in dataset.
        :param str folder_name: Directory containing all image/xml files. Defaults to 'data'.
        :param int max_images_per_class: Maximum number of images with annotations to use in training.
        :param bool train: If True, will perform data augmentation on images for training.
        """
        self.folder_name = folder_name
        self.obj_class_labels = obj_class_labels
        self.train = train

        imgs = []
        xml_files = []
        # iterate through object classes, loading data from folders, and create list of image and xml file names
        for obj_class in obj_class_labels:
            obj_class = obj_class.lower()

            class_imgs = os.listdir(os.path.join(folder_name, obj_class, 'images'))
            class_img_filepaths = [os.path.join(folder_name, f'{obj_class}/images/{i}') for i in class_imgs]
            imgs.extend(class_img_filepaths[:max_images_per_class])

            class_xml_files = os.listdir(os.path.join(folder_name, obj_class, 'xml_files'))
            class_xml_filepaths = [os.path.join(folder_name, f'{obj_class}/xml_files/{i}') for i in class_xml_files]
            xml_files.extend(class_xml_filepaths[:max_images_per_class])

        # sort before assignment so that corresponding images and xml files have the same indices in each list
        self.imgs = list(sorted(imgs))
        self.xml_files = list(sorted(xml_files))

    @staticmethod
    def parse_xml(filename):
        """
        Parses .xml file, extracting all ground truth boundary box coordinates with corresponding labels.

        :param str filename: .xml file
        :return: Tuple containing a list of each set of boundary box coordinates and a list of corresponding labels.
        :rtype: tuple
        """
        # read file and convert to parse-able format
        xml_contents = open(filename).read()
        soup = BeautifulSoup(xml_contents, 'lxml')

        boxes = []
        labels = []
        # get each boundary box specified in file
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
        """
        Runs when using indexing behavior ([]) on this class.

        :param int idx: Index of image/xml file pair to get.
        :return: Tuple containing image and target (basically metadata for images needed for training the model).
        :rtype: tuple
        """
        # load images and annotations
        img_path = self.imgs[idx]
        xml_path = self.xml_files[idx]

        # open image and parse xml file
        img = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_xml(xml_path)

        # prepare target data
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        label_idxs = [self.obj_class_labels.index(label) + 1 for label in labels]

        # transformations
        img_transform_list = [T.ToTensor()]
        if self.train:
            # Flip image and boundary box with 50% chance
            if random.uniform(0, 1) > 0.5:
                img, boxes = self.flip(img, boxes)

        # perform transforms on image
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
