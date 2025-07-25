Languages: [中文](README.md) | [English](README_EN.md)

# Project Overview

This project is based on YOLOv8 and introduces several architectural enhancements to improve its performance in small object detection scenarios. The improvements include:

* Adding a small object detection head
* Introducing Omni-Dimensional Dynamic Convolution (ODConv)
* Using Depthwise Separable Convolution (DWConv)
* Adding a Multi-Head Self-Attention (MHSA) module


# Environment Setup & Dependencies

## Required Environment

| Software            | Recommended Version | Notes                                                 |
| ------------------- | ------------------- | ----------------------------------------------------- |
| Python              | 3.9.13              | Please check "Add Python to PATH" during installation |
| Anaconda (Optional) | 2024.06             | Optional but recommended                              |
| CUDA                | 11.7.1              |                                                       |
| PyTorch             | 1.13.1+cu117        |                                                       |

## Upgrade pip

```bash
python -m pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple
```

## Install PyTorch with CUDA support

Using PyTorch 1.13.1 + CUDA 11.7 as an example:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
-i https://pypi.tuna.tsinghua.edu.cn/simple \
--extra-index-url https://download.pytorch.org/whl/cu117
```

After installation, run `pip list` to verify the versions.

## Install Project Dependencies

Navigate to the project directory and run:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> If you're using Anaconda, first run:
>
> ```bash
> conda create -n yolov8-improve python=3.9
> conda activate yolov8-improve
> ```
>
> Then continue with the pip installation as above.

# Project Structure

```
/Faster_Rcnn/
    └── train.py                # Training script for Faster R-CNN
/YoloV5/
    └── train.py                # Training script for YOLOv5
/YoloV8/
    └── yoloV8-train.py         # Original YOLOv8 training script
/YoloV8_Improve/                # Improved YOLOv8 model and configs
    ├── yoloV8-train.py         # Training script for the improved YOLOv8
    ├── train_yaml/             # Improved model configuration files
    └── ultralytics/nn/
        ├── modules/odconv.py   # ODConv: Omni-Dimensional Dynamic Convolution
        ├── modules/conv.py     # DWConv: Depthwise Separable Convolution
        └── MHSA.py             # Multi-Head Self-Attention module
/DataSet/                       # Data processing scripts
    ├── txt_to_xml.py           # Convert YOLO format to VOC
    ├── xml_to_txt.py           # Convert VOC format to YOLO
    └── datasetPartitioning.py  # Dataset splitting script
```

# Model Training

Place your dataset into the `all_images`, `all_txt`, and `all_xml` folders under the `DataSet` directory.
Then run `datasetPartitioning.py` to split the dataset into training and validation sets.
Finally, use the appropriate training script to start model training.


# Further Improvements

This project does not modify the base YOLOv8 structure beyond the improvements mentioned.
To customize data directories, adjust model configurations, or add/remove modules, refer to the official YOLOv8 documentation and modify the scripts accordingly.

# References

This project was developed as part of a graduation thesis focusing on enhancing small object detection in YOLOv8.
All academic resources and implementations referenced are listed in the `REFERENCES.md` file.

> This project is intended for academic and research purposes only.
> Some parts are inspired by existing research papers and public repositories.
> If any rights are infringed, please contact us for removal.

---
