#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from lxml.etree import Element, SubElement, tostring, ElementTree

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from PIL import Image
import numpy as np

classes = ['Car', 'Large_Truck'] # 类别

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    w = np.clip(w, 0, 1)
    h = np.clip(h, 0, 1)
    return (x, y, w, h)


def convert_annotation(image_id, xml_path, txt_path):
    in_file = open(xml_path + '/%s.xml' % (image_id), encoding='UTF-8')

    out_file = open(txt_path + '/%s.txt' % (image_id), 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    # size = root.find('size')
    #
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    img_path = './all_images/' + str(image_id) + ".jpg"
    img = Image.open(img_path)
    w = img.width  # 图片的宽
    h = img.height  # 图片的高

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # print(cls)
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    my_xml_path = './all_xml'
    my_txt_path = './all_txt'
    if not os.path.exists(my_txt_path):
        os.mkdir(my_txt_path)

    xml_path = os.path.join(CURRENT_DIR, './all_xml/')

    # xml list
    img_xmls = os.listdir(xml_path)
    for img_xml in img_xmls:
        label_name = img_xml.split('.')[0]
        print(label_name)
        convert_annotation(label_name,my_xml_path,my_txt_path)
