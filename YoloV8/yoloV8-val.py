from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./runs/detect/train/weights/best.pt')
    model.val(split='test', data='./my_data.yaml', workers=4, batch=16)
