from ultralytics import YOLO


if __name__ == '__main__':
    final_path = 'train_yaml/yolov8n.yaml'
    model = YOLO(final_path)
    model.train(data='./my_data.yaml', workers=8, epochs=200, batch=32, optimizer='SGD', device='0')
