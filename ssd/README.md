# SSD model
**This a master final project of X-Ray object detection. All code is based on [SSD](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD).**

## Configure Environment：
* Python == 3.6 or 3.7
* Pytorch == 1.5
* pycocotools (Linux: pip install pycocotools;
  Windows:pip install pycocotools-windows)
* Ubuntu or Centos (Windows not recommanded)
* At best training by GPU

## File Structure：
```
├── src:                         Implement relevant modules of SSD model.
│     ├── resnet50_backbone.py:  Use resnet50 network as SSD backbone
│     ├── ssd_model.py:          SSD network structure file
│     └── utils.py:              Some of the functions used in the training process
├── train_utils:                 Training and validation related modules (including cocotools)
├── my_dataset.py:               Custom dataset is used to read X-Ray dataset
├── train_ssd300.py:             Training SSD model based on resnet50 as backbone
├── train_multi_GPU.py:          For users with multiple GPUs 
├── predict_test.py:             Simple prediction script, using trained weights for prediction tests
├── XRay_classes.json:           X-Ray data categories
├── plot_curve.py:               Map to mAP the losses of the training process and the validation set
├── instruction.md:              The cover page designed for web application
├── micron_detection.py:         Streamlit app developed for data visualization, run by command 'sreamlit run micron_detection.py'
```

## Download address of pre-training weights: 
  ### (put pre-training weights into src folder after downloading)
**The well-trained weights can be used for both testing and training directly.**
* The well-trained weights (could be used for testing directly) can be downloaded from:\
https://pan.baidu.com/s/1HOtP4dp3Zxt50fyrofxB5w password：kw1z

**The official weights can only be used for trainging, not for testing.**
* The official weights of ResNet50+SSD could be found: https://ngc.nvidia.com/catalog/models \
 `search ssd -> find SSD for PyTorch(FP32) -> download FP32 -> unzip`
* The official weights also could be downloaded from Baidu Netdisk:\
  https://pan.baidu.com/s/1byOnoNuqmBLZMDA0-lbCMQ password: iggj

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
* Single GPU training or CPU, directly use train_ssd300.py training script.

## Testing Method:
* Make sure to modify prediction path correctly in 'predict_test.py'.
* Runing 'predict_test.py'

## Questions? Problems?
`If any problems or questions, please feel free to contact email: e0427741@u.nus.edu`
