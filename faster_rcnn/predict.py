import torch
import torchvision
from torchvision import transforms
#from my_dataset import XRayDataset
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
#from network_files.rpn_function import AnchorsGenerator
#from backbone.mobilenetv2_model import MobileNetV2
from draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
import os

def create_model(num_classes):
    '''
        This is a function to create a basic faster RCNN model
    '''
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    return model


# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# create model
model = create_model(num_classes=6)

# load train weights
train_weights = "./save_weights/resNetFpn-model-21.pth"
model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
#model.to(device)

# read class_indict
category_index = {}
try:
    json_file = open('./XRay_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

if not os.path.exists('F:/5005_object_detection/baseline/faster_rcnn_few_shot_learning/predict/'):
    os.makedirs('F:/5005_object_detection/baseline/faster_rcnn_few_shot_learning/predict/')
# load image
img_path = 'F:/5005_object_detection/baseline/faster_rcnn_few_shot_learning/train_data/val/Images/'
for i in os.listdir(img_path):
    original_img = Image.open(os.path.join(img_path, i))

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        predictions = model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to(device).numpy()
        predict_classes = predictions["labels"].to(device).numpy()
        predict_scores = predictions["scores"].to(device).numpy()
        #print(predict_scores)

        if len(predict_boxes) == 0:
	    #original_img.save('/home/svu/e0427741/predictions/' + i)
	    #continue
            print("Detected nothing in this image!")
        #print('predict_boxes:\n', predict_boxes)
        #print('predict_classes:\n', predict_classes)
        #print('predict_scores:\n', predict_scores)
        
        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.2,
                 line_thickness=5)
        #plt.imshow(original_img)
        #plt.show()
        original_img.save('F:/5005_object_detection/baseline/faster_rcnn_few_shot_learning/predict/' + i)
