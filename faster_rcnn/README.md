# Faster R-CNN
**This a master final project of X-Ray object detection. All code is based on [Faster_RCNN](https://pytorch.org/docs/stable/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn).**

## Configure Environment：
* Python == 3.6 or 3.7
* Pytorch == 1.5
* pycocotools (Linux: pip install pycocotools;
  Windows:pip install pycocotools-windows)
* Ubuntu or Centos (Windows not recommanded)
* At best training by GPU

## File Structure：
```
├── backbone:                          Feature extraction network can be selected according to their own requirements.
│      ├── mobilenetv2_model.py:       Mobile Net as backbone.
│      ├── resnet50_fpn_model.py:      Resnet50 as backbone.
│      └── vgg_model.py:               VGG as backbone.
├── network_files:                     Faster R-CNN network structure file
│      ├── boxes.py:                   Caculate boxes IoU.
│      ├── det_utils.py:               Balance positive and negative samples.
│      ├── faster_rcnn_framework.py:   Main framework of faster RCNN.
│      ├── image_list.py:              List out images.
│      ├── roi_head.py:                ROI head.
│      ├── rpn_function.py:            RPN network.
│      └── transform.py:               Data transform.
├── train_utils:                       Training and validation related modules (including cocotools).
│      ├── coco_eval.py:               
│      ├── coco_utils.py:              
│      ├── group_by_aspect_ratio.py:   
│      └── train_eval_utils.py:        
├── my_dataset.py:                     Custom dataset is used to read X-Ray dataset.
├── train_mobilenet.py:                Training Faster RCNN model based on mobilenet as backbone.
├── train_resnet50_fpn.py:             Training Faster RCNN model based on resnet50 as backbone.
├── train_multi_GPU.py:                For users with multiple GPUs.
├── predict.py:                        Simple prediction script, using trained weights for prediction tests.
├── XRay_classes.json:                 X-Ray data categories.
├── instruction.md:                    The cover page designed for web application.
├── micron_detection.py:               Streamlit app developed for data visualization, run by command 'sreamlit run micron_detection.py'.
```

## Download address of pre-training weights: 
  ### (put pre-training weights into save_weights folder after downloading)
**The well-trained weights can be used for both testing and training directly.**
* The well-trained weights of ResNet50+FPN (could be used for testing directly) can be downloaded from:\
  https://pan.baidu.com/s/1CCIANv-c93YRiENcOLDrBQ password：6h7j 

**The official weights can only be used for trainging, not for testing.**
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

## Dataset (place data in the current folder of the project, named as train_data):
**The train_data folder structure:**
```
├── train_data: 
│     ├── train:              Training Data
│     │     ├── Images:       Training Images (.BMP or .JPG)
│     │     └── Annotations:  Training Annotation (.xml)
│     └──  val:               Validation Data    
│           ├── Images:       Validation Images (.BMP or .JPG)
│           └── Annotations:  Validation Annotations (.xml)
```

## Training Method:
* Be sure to prepare your data set in advance;
* Make sure to download the corresponding pre-trained model weights in advance;
* Single GPU training or CPU, directly use train_res50_fpn.py training script.

## Testing Method:
* Make sure to modify weights path correctly in 'predict.py'.
* Runing 'predict.py'

## Questions? Problems?
`If any problems or questions, please feel free to contact email: e0427741@u.nus.edu`
