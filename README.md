语言: [中文](README.md) | [English](README_EN.md)

# 项目说明
本项目以YoloV8为基础，通过添加小目标检测头分支、引入全维度动态卷积ODConv和深度可分离卷积DWConv、添加全局多头自注意力机制MHSA模块的四种方式，针对小目标识别场景对YoloV8的架构进行了一定的优化。

> 注意：这个项目是本科毕业设计的一部分，水平仅限于本科阶段，没有太多实用价值。而且因为教育改革的规定，源代码已经随论文提交到统一数据库。如果直接照搬，可能会因为和过往作品相似度过高而被抽查或举报（即便你所在的省份目前还没有相关规定，但随着改革的进行，届时往届生也会纳入抽查范围）。所以如果你打算把这个项目当作自己的毕业设计，请一定进行多项修改，若要照搬则务必慎重考虑后果，我无法替你负责。

# 环境配置与依赖安装

## 所需环境

| 软件名称                                 | 推荐版本         | 特殊说明                     |
| ------------------------------------ | ------------ | -------------------------- |
| Python                               | 3.9.13       | 安装时请勾选“Add Python to PATH” |
| Anaconda                             | 2024.06      | 也可以不用                        |
| CUDA                                 | 11.7.1       |                                 |
| PyTorch                              | 1.13.1+cu117 |                               |


## 升级 pip
```
python -m pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 安装 PyTorch 和 CUDA 支持
以PyTorch1.13.1+cu117为例：
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
-i https://pypi.tuna.tsinghua.edu.cn/simple \
--extra-index-url https://download.pytorch.org/whl/cu117
```
安装完成后可以使用 `pip list` 验证版本。

## 安装项目依赖
cd进入项目目录并执行以下命令：
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 如果使用 Anaconda ，则先执行：
> ```
> conda create -n yolov8-improve python=3.9
> conda activate yolov8-improve
> ```
> 然后再执行上面 pip 的安装步骤。


# 项目结构说明

```
/Faster_Rcnn/
    └── train.py               # Faster R-CNN训练脚本
/YoloV5/
    └── train.py               # YOLOv5训练脚本
/YoloV8/
    └── yoloV8-train.py        # 原版YOLOv8训练脚本
/YoloV8_Improve/               # 改进后的YOLOv8模型及配置文件
    ├── yoloV8-train.py        # 改进型YOLOv8训练脚本
    ├── train_yaml/            # 改进版模型结构配置
    └── ultralytics/nn/
        ├── modules/odconv.py  # 全维度动态卷积 ODConv
        ├── modules/conv.py    # DWConv 深度可分离卷积
        └── MHSA.py            # 多头自注意力模块
/DataSet/                      # 数据处理相关脚本
    ├── txt_to_xml.py          # YOLO格式转VOC
    ├── xml_to_txt.py          # VOC转YOLO格式
    └── datasetPartitioning.py # 数据集划分脚本
```

# 训练模型

在 `DataSet` 的 `all_images` 、 `all_txt` 、 `all_xml` 中放入你的数据集，然后用 `datasetPartitioning.py` 划分数据集，最后对应的训练脚本训练即可。

# 进一步改进

本项目未对模型结构进行修改，如果要指定新的数据集目录或修改配置、增减模块，参考原版YoloV8的文档修改脚本配置即可。

# 参考资料

这个项目是毕业设计的一部分，课题是研究如何提升YoloV8的小目标识别能力，其中参考的学术资料在 `REFERENCES.md` 中完整列出。

This project was developed as part of a graduation thesis focused on improving the YOLOv8 object detection framework. The following academic works and implementations were referenced during the research and development process:

Please refer to `REFERENCES.md` for the full list of academic sources consulted.

> This project is for academic and research purposes only. Some parts are inspired by existing research papers and public implementations. If any rights are infringed, please contact for removal.
