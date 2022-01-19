# 🕹 CUBOX Image Classification
> ***Image Classification with CUBOX dataset***
>
>
> Related Paper
> https://arxiv.org/pdf/1512.03385.pdf

#### ️ Maintainer
 - [Sunoh Lee](https://github.com/sunohlee)
 - [Yura Choi](https://github.com/Yuuraa)
 

---
### 📌 **Table Of Contents**

- [Experiment Environment](#-experiment-environment)
    - [Hardware](#-hardware)
    - [Setup](#-setup)
    - [Dataset folder structure](#--dataset-folder-structure)
- [Scripts](#-scripts)
    - [Train Script](#1-train-script)
    - [Test Script](#2-test-script)
      - [Test result folder structure]()
    
---

### 💻 **Experiment Environment**
<br/>

#### **⚙️ Software**

Tested on:
- [Ubuntu 18.04](https://ubuntu.com/)
- [Python 3.6.9](https://www.python.org/)
- [NVIDIA Driver 460.80](https://www.nvidia.com/Download/index.aspx)
<br/>

#### **:whale:Docker Image**
yoorachoi/cubox_cls:resnet

#### **⚙️ Hardware**

Tested on:
    - **GPU** - 1 x TITAN V (12GB)
    - **CPU** - Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz  
    - **RAM** - 64GB
    - **SSD** - 1TB

<br/>

#### **⛳ Setup**


--------------------------

```bash
# making own virtual environment
pip install virtualenv
python3 -m venv venv
```
<br/>

```bash
# activating own virtual environment
source ./venv/bin/activate
pip install -r requirement.txt # install some packages in venv with requirement.txt

pip freeze #check installed package in own virtual environment(venv)
deactivate # exit venv
```

<br/>

####  **📁  Dataset folder Structure**

```
./
└── long_tail_total_images_first_cubox_sub
   ├── train
   |   ├── none
   |   |   ├── class1
   |   |   |   ├── image1
   |   |   |   └── image2
   |   |   |   └── ...
   |   |   └── class2
   |   |   └── class3
   |   |   └── ...
   |   └── semitransparent
   |   |   ├── class1
   |   |   └── class2
   |   |   └── class3
   |   |   └── ...
   |   └── wiredense
   |   |   ├── class1
   |   |   └── class2
   |   |   └── class3
   |   |   └── ...
   |   └── wireloose
   |   |   ├── class1
   |   |   └── class2
   |   |   └── class3
   |   |   └── ...
   |   └── wiremedium
   |   |   ├── class1
   |   |   └── class2
   |   |   └── class3
   |   |   └── ...
   └── validation # same structure as train
   └── test        # same structure as train

```



<br/>

## 📜 Scripts
<br/>

### 1️⃣ Train Script

Command format:
```
CUDA_VISIBLE_DEVICES=0 python main_CNN.py # default occlusion type = ['none']

CUDA_VISIBLE_DEVICES=0 python main_CNN.py --training_occlusion_type "none" "semitransparent" "wiredense" "wiremedium" "wireloose"
```


<br/>

### 2️⃣ Test Script

Command format:
```
CUDA_VISIBLE_DEVICES=0 python just_test_longtail.py # a lot of occlusion type combination testing
```

```
#default arguments
test_occlusion_list = [['none'], ['semitransparent'], ["wiredense"], ["wiremedium"], ["wireloose"],  
                       ['semitransparent', "wiredense", "wiremedium", "wireloose"],  
                       ['none', 'semitransparent', "wiredense", "wiremedium", "wireloose"]]
```
<br/>


After Test, We can get accuracy txt files, confusion matrixes, sample distributions per class. details are in below 

####  **📁  Test result folder structure**
```
./
└── total_case_lt_train_weight
   ├── model_weights_LR_001_step_size_30_training_with_None_longtail
   |   ├── occlusion_type_with_None    # testing occlusion type
   |   |   ├── split_with_Train
   |   |   |   ├── confusion_matrix.png
   |   |   |   └── distribution_of_class.png
   |   |   |   └── test_result.txt
   |   |   |
   |   |   ├── split_with_Validation
   |   |   └── split_with_Test
   |   |
   |   ├── same structure with another testing occlusion type
   |   ├── config.txt      # few training configs
   |   ├── distribution_of_class.png     # training dataset's sample distribution
   |   ├── model_weights_44_LR_001_step_size_30_training_with_None_best_acc_per_class_mean.pth   # first number is epoch, best acc weight file
   |   └──  ... # weight file per 5epoch
   |
   ├── same structure with another training occlusion type
   ├─- .....
   └── total_result_lt.txt
       
```