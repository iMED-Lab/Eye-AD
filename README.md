<!--
 * @Author: JinkuiH jinkui7788@gmail.com
 * @Date: 2023-07-13 12:34:24
 * @LastEditors: JinkuiH jinkui7788@gmail.com
 * @LastEditTime: 2023-07-13 15:04:17
 * @FilePath: \Eye-AD\README.md
 * @Description: 
 * 
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
-->
# Eye-AD
Eye-AD: A multilevel graph-based model for Alzheimer’s Disease detection and risk assessment based on retinal OCTA images

## Introduction
![Eye-AD](img\Eye-AD.png)

We design a novel multilevel graph representation to formulate and mine the intra-instance and inter-instance relationship of multiple en face projections acquired by OCTA devices, including superficial vascular complex, deep vascular complex, and choriocapillaris.

## Getting Started

- Create a new  environment:
```bash
conda create -n Eye-AD python=3.8
```

- Activate the environment:
```bash
conda activate Eye-AD
```

- Clone the repository from GitHub:
```bash
git clone https://github.com/iMED-Lab/Eye-AD.git
```

- Install prerequisites

```bash
cd Eye-AD
pip install -r requirements.txt
```

### Prepare your data

Please put the root directory of your dataset into the folder ./data. The root directory contain the two subfolder now: AD and control. The most convenient way is to follow the sample file structure, as follows:

```
|-- .data
    |-- root directory
        |-- AD
        |-- control
            |-- ID_name
                |-- macular3_3
                    |-- __SVC.png
                    |-- __DVC.png
                    |-- __choriocapillaris.png
                    |-- ... 
```

You can also change the file structure. Note that you need to change the data processing function (i.e., __prepare() at line 118 in test.py) to ensure the data can be obtained correctly. 

Due to the method need the multiple inputs, i.e., SVC, DVC and choriocapillaris, so the most important thing is that you need specify the filter words for file name of SVC, DVC, and  choriocapillaris at line 124 in test.py. Please make sure the three filter words are in the right order.

### Start training
You can change the experiment parameters by modifying the configuration file and then come to train the model.

```
python train.py
```

### Start evaluation

```
python test.py
```
The results will be automatically saved in the . /results folder.