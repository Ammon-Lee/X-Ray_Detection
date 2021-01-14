from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
#import cv2
from lxml import etree


class XRayDataset(Dataset):
    """Read X-Ray images"""

    def __init__(self, root = './', transforms = None, train_set = True):
        self.root = os.path.join(root, "train_data")
        '''
        self.img_root = os.path.join(self.root, 'Images')
        self.xml_root = os.path.join(self.root, 'Annotations')
        self.xml_list = os.listdir(self.xml_root)
        '''
        # the image and annotations root of train and validation data.
        if train_set:
            self.img_root = os.path.join(self.root, "train", "Images")
            self.annotations_root = os.path.join(self.root, "train", "Annotations")
        else:
            self.img_root = os.path.join(self.root, "val", "Images")
            self.annotations_root = os.path.join(self.root, "val", "Annotations")
        
        self.xml_list = os.listdir(self.annotations_root)
        
        # read class_indict
        try:
            json_file = open('./XRay_classes.json', 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = os.path.join(self.annotations_root, self.xml_list[idx])
        # xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        height_width = [data_height, data_width]
        img_path = os.path.join(self.img_root, data["filename"])
        #print(data["filename"])
        image = Image.open(img_path)
        '''
        if image.format != "JPEG":
            raise ValueError("Image format not JPEG")
        '''
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"]) / data_width
            xmax = float(obj["bndbox"]["xmax"]) / data_width
            ymin = float(obj["bndbox"]["ymin"]) / data_height
            ymax = float(obj["bndbox"]["ymax"]) / data_height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["height_width"] = height_width

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        Transform xml files to dictionary.
        Args：
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result: 
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}


# import transforms
# from draw_box_utils import draw_box
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = VOC2012DataSet(os.getcwd(), data_transform["train"], True)
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()
