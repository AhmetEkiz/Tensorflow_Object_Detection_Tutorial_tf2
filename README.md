# README

This is my Tensorflow Object Detection API folder to train and create interence repository.

## Requirements

- [Tensorflow Object Detection API](https://github.com/tensorflow/models)

- | Prerequisites                 |
  | ----------------------------- |
  | Nvidia GPU (GTX 650 or newer) |
  | CUDA Toolkit v11.2            |
  | CuDNN 8.1.0                   |

## Setup Environment

```bash
# to create new environment
conda env create -f environment.yml

# to activate environment
conda activate tf_od 

# to add conda environment as Jupyter Notebook kernel
# you should run from base environment that installed jupyter notebook
conda install pip  # if you don't intsall it with environment.yml file
python -m ipykernel install --user --name=tf_od
```

```bash
# If you use Windows and want to use Unix-based commands with Anaconda
 conda install m2-base
```

## Commands

```bash
# to run real time object detection example
python real_time_inference.py

# to run webcam overview to compare fps 
python webcam.py
```

## Files and Folders

- **models :** that contains Tensorflow Object Detection API 

- **workspace :** the main folder that contains example images, videos and pre-trained models

- **object_detection_tutorial.ipynb :** Starter Notebook to test and review TF OD API that works.

## First Time Installing and Errors

- Download **Visual Studio Community** and **Desktop Development with C++**
  
  - This is requirement for **COCO API**
  
  - You should install latest Protoc but, it needs to be compatible with installed protoc version. I say this beacause it can gives you error:

> [python - ImportError: cannot import name &#39;builder&#39; from &#39;google.protobuf.internal&#39; - Stack Overflow](https://stackoverflow.com/a/74237049/8654414)
> 
> I use Anaconda Prompt and Environment on Windows 11.
> 
> I solved the problem by making the same version of the two Protobuff 
> installs one is the Anaconda Protobuff install, and the other one is 
> that I installed from [Releases · protocolbuffers/protobuf · GitHub](https://github.com/protocolbuffers/protobuf/releases)
> 
> In order to make the same version, I reinstall Protobuff releases that are compatible with the Anaconda Protoc installation.
> 
> You can see what is your Protobuf with `pip list`

- **OpenCV Installs Errors:**
  
  - [python - AttributeError: partially initialized module &#39;cv2&#39; has no attribute &#39;gapi_wip_gst_GStreamerPipeline&#39; (most likely due to a circular import) - Stack Overflow](https://stackoverflow.com/a/74245841/8654414)
  
  - **uninstall** all **opencv related libraries** and **install** just **opencv-python**

```bash
pip uninstall opencv-contrib-python
pip uninstall opencv-python
pip uninstall opencv-python-headless

pip install opencv-python
```

## To-do

- [ ] Export TF Lite version of Models

- [ ] Training process

# 
