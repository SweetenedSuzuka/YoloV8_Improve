import cv2
import numpy as np
from PIL import Image, ImageDraw


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
    line    = annotation_line.split()
    #------------------------------#
    #   读取图像并转换成RGB图像
    #------------------------------#
    image   = Image.open(line[0])
    image   = image.convert('RGB')

    #------------------------------#
    #   获得图像的高宽与目标高宽
    #------------------------------#
    iw, ih  = image.size
    h, w    = input_shape
    #------------------------------#
    #   获得预测框
    #------------------------------#
    box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        #---------------------------------#
        #   将图像多余的部分加上灰条
        #---------------------------------#
        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data  = np.array(new_image, np.float32)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        return image_data, box
            
    #------------------------------------------#
    #   对图像进行缩放并且进行长和宽的扭曲
    #------------------------------------------#
    new_ar = iw/ih * rand(1-jitter,1+jitter) / rand(1-jitter,1+jitter)
    scale = rand(.8, 3)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    #------------------------------------------#
    #   将图像多余的部分加上灰条
    #------------------------------------------#
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    #------------------------------------------#
    #   翻转图像
    #------------------------------------------#
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image_data      = np.array(image, np.uint8)
    #---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    #---------------------------------#
    r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype           = image_data.dtype
    #---------------------------------#
    #   应用变换
    #---------------------------------#
    x       = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    #---------------------------------#
    #   对真实框进行调整
    #---------------------------------#
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] 
    
    return image_data, box

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def get_random_data_with_Mosaic(annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
    h, w = input_shape
    min_offset_x = rand(0.3, 0.7)
    min_offset_y = rand(0.3, 0.7)

    image_datas = [] 
    box_datas   = []
    index       = 0
    for line in annotation_line:
        #---------------------------------#
        #   每一行进行分割
        #---------------------------------#
        line_content = line.split()
        #---------------------------------#
        #   打开图片
        #---------------------------------#
        image = Image.open(line_content[0])
        image = image.convert('RGB')
        
        #---------------------------------#
        #   图片的大小
        #---------------------------------#
        iw, ih = image.size
        #---------------------------------#
        #   保存框的位置
        #---------------------------------#
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
        
        #---------------------------------#
        #   是否翻转图片
        #---------------------------------#
        flip = rand()<.5
        if flip and len(box)>0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0,2]] = iw - box[:, [2,0]]

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * rand(1-jitter,1+jitter) / rand(1-jitter,1+jitter)
        scale = rand(.8, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        #-----------------------------------------------#
        #   将图片进行放置，分别对应四张分割图片的位置
        #-----------------------------------------------#
        if index == 0:
            dx = int(w*min_offset_x) - nw
            dy = int(h*min_offset_y) - nh
        elif index == 1:
            dx = int(w*min_offset_x) - nw
            dy = int(h*min_offset_y)
        elif index == 2:
            dx = int(w*min_offset_x)
            dy = int(h*min_offset_y)
        elif index == 3:
            dx = int(w*min_offset_x)
            dy = int(h*min_offset_y) - nh
        
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)

        index = index + 1
        box_data = []
        #---------------------------------#
        #   对box进行重新处理
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        
        image_datas.append(image_data)
        box_datas.append(box_data)

    #---------------------------------#
    #   将图片分割，放在一起
    #---------------------------------#
    cutx = int(w * min_offset_x)
    cuty = int(h * min_offset_y)

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    new_image       = np.array(new_image, np.uint8)
    #---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    #---------------------------------#
    r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
    dtype           = new_image.dtype
    #---------------------------------#
    #   应用变换
    #---------------------------------#
    x       = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

    #---------------------------------#
    #   对框进行进一步的处理
    #---------------------------------#
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes

def get_random_data_with_MixUp(image_1, box_1, image_2, box_2):
    new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
    if len(box_1) == 0:
        new_boxes = box_2
    elif len(box_2) == 0:
        new_boxes = box_1
    else:
        new_boxes = np.concatenate([box_1, box_2], axis=0)
    return new_image, new_boxes
