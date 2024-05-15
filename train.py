from ultralytics import YOLO

model=YOLO('runs/detect/train2/weights/best.pt')

# res=model.train(data='datasets/data.yaml',epochs=500,workers=0,batch=64,imgsz=224)


model.export(format='onnx')