# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, cv2
from PIL import Image

import torch
import torchvision
from torchvision import transforms as t

from lxml import etree
import json

from draw_box_utils import draw_box
from src.ssd_model import SSD300, Backbone

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))
    
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Select Mode")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Instructions", "X-ray Detection"])
    if app_mode == "Instructions":
        st.sidebar.success('To do detection, please choose "X-ray image detection".')
    elif app_mode == "X-ray Detection":
        readme_text.empty()
        run_the_app()

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # First of all, read the test dataset list.
    @st.cache
    def data_list(val_data_path):
        return os.listdir(val_data_path)

    xml_list = data_list(val_data_path)

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    xml, xml_index = frame_selector(xml_list)
    if xml_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return
    
    # read xml file.
    xml_data = get_file_content_as_dict(xml)

    # get the category index.
    category_index, class_dict = category_dict()

    # the ground truth label of xml_data
    boxes_gt, classes_gt = get_ground_truth_from_xml(xml_data, class_dict)
    #scores_gt = [1.0] * len(classes_gt)
    
    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold = object_detector()
    
    # Load the image.
    image_path = os.path.join(val_image_path, xml_data['annotation']["filename"])
    image_pil, image = load_image(image_path)
    
    # Add boxes for objects on the image. These are the boxes for the ground image.
    # boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    
    draw_image_with_boxes(image,
                          boxes_gt,
                          classes_gt, 
                          "Ground Truth",
                          "**Human-annotated data** (frame `%i`)" % xml_index)
    
    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    predict_boxes, predict_classes, predict_scores = ssd(image=image_pil,
                                                         weights_path=weights_path)

    predict_boxes, predict_classes = select_items_above_confidence(predict_boxes,
                                                                   predict_classes,
                                                                   predict_scores,
                                                                   confidence_threshold)

    #print(predict_scores)
    draw_image_with_boxes(image,
                          predict_boxes,
                          predict_classes,
                          "Real-time Computer Vision",
                          "**SSD Model** (confidence `%3.1f`)" % confidence_threshold)

def parse_xml_to_dict(xml):
    """
    Transform xml contents to python dictionary.
    Refering to the code of recursive_parse_xml_to_dict of tensorflow.
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # Traversal to the bottom, directly return the tag corresponding information
        return {xml.tag: xml.text}

    # Recursively traverses the label information
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

# This sidebar UI is a little search engine to find certain object types.
def frame_selector(xml_list):
    st.sidebar.markdown("# Frame")

    # Choose a frame out of all frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(xml_list)-1, 0)

    selected_frame = xml_list[selected_frame_index]
    return selected_frame, selected_frame_index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    # line_thickness = st.sidebar.slider('Line thickness', 1.0, 10.0, 5.0, 0.5)
    return confidence_threshold #, line_thickness

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, classes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = {
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255],
        4: [255, 255, 0],
        5: [255, 0, 255],
        }
    
    image_with_boxes = image.astype(np.float64)
    for label, (xmin, ymin, xmax, ymax) in zip(classes, boxes):
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] /= 2

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

def select_items_above_confidence(boxes, labels, scores, confidence):
    selected_filter = scores >= confidence
    return boxes[selected_filter], labels[selected_filter]

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    path = os.path.join('./', path)
    f = open(path, 'r')
    return f.read() #.decode("utf-8")

# read xml data and transform data to a dictionary.
def get_file_content_as_dict(xml):
    xml_path = os.path.join(val_data_path, xml)
    with open(xml_path) as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    xml_data = parse_xml_to_dict(xml)
    return xml_data

def get_ground_truth_from_xml(xml_data, class_dict):
    boxes_gt, classes_gt = [], []
    
    for obj in xml_data['annotation']["object"]:
        if obj == "\n\t":
            continue
        xmin = float(obj["bndbox"]["xmin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymin = float(obj["bndbox"]["ymin"])
        ymax = float(obj["bndbox"]["ymax"])
        boxes_gt.append([xmin, ymin, xmax, ymax])
        classes_gt.append(class_dict[obj["name"]])
    return np.array(boxes_gt), np.array(classes_gt)

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image(image_path):
    image_pil = Image.open(image_path)
    image = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
    image = image[:, :, [2, 1, 0]]
    return image_pil, image

@st.cache
def category_dict():
    category_index = {}
    try:
        json_file = open('./XRay_classes.json', 'r')
        class_dict = json.load(json_file)
        category_index = {v:k  for k, v in class_dict.items()}
        class_dict = {k:v for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)
    return category_index, class_dict

# Load the network.
@st.cache(allow_output_mutation=True)
def load_network(weights_path, device, num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)
    #net = create_model(num_classes=6)
    model.load_state_dict(torch.load(weights_path, map_location=device)["model"])
    return model

# Run the Faster RCNN model to detect objects.
def ssd(image, weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # read class dictionary.
    category_index = category_dict()


    data_transform = t.Compose([t.Resize((300, 300)),
                                t.ToTensor(),
                                t.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
  
    img = data_transform(image)

    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # predict process by faster RCNN
    model = load_network(weights_path=weights_path, device=device, num_classes=6)

    model.eval()
    with torch.no_grad():
        predictions = model(img.to(device))[0]
        predict_boxes = predictions[0].to(device).numpy()
        predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * image.size[0]
        predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * image.size[1]
        predict_classes = predictions[1].to(device).numpy()
        predict_scores = predictions[2].to(device).numpy()

    if len(predict_boxes) == 0:
        print('Detected nothing in this image!')

    return predict_boxes, predict_classes, predict_scores

# define the global path to validation data and weights path.
weights_path = './save_weights/ssd300-23.pth'
path = './train_data/val'

val_data_path = os.path.join(path, 'Annotations')
val_image_path = os.path.join(path, 'Images')

if __name__ == "__main__":
    main()
