from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
# import cv2
from lxml import etree


class XRayDataset(Dataset):
    '''
    Read X-Ray images and corresponding annotations.


    The format of images and annotations:
    images: BMP flies,
    annotations: XML files.
    '''
    
    def __init__(self, root, transform, train_set=True):
        self.root = os.path.join(root, "train_data")
        if train_set:
            self.img_root = os.path.join(self.root, "train", "Images")
            self.annotations_root = os.path.join(self.root, "train", "Annotations")
        else:
            self.img_root = os.path.join(self.root, "val", "Images")
            self.annotations_root = os.path.join(self.root, "val", "Annotations")
        
        # read class_indict
        try:
            json_file = open('./XRay_classes.json', 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        self.xml_list = os.listdir(self.annotations_root)
        self.transform = transform

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = os.path.join(self.annotations_root, self.xml_list[idx])
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        #print("data: ", data)
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        '''
        if image.format != "JPEG":
            raise ValueError("Image format not JPEG")
        '''
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            if obj == '\n\t':
                continue
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target)

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
        Transform xml contents to python dictionary.
        Refering to the code of recursive_parse_xml_to_dict of tensorflow.
        Args：
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}
