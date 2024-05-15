import math
import time
import kf2d
import cv2
import numpy as np
import fastdeploy as fd
import draw_key_point

op = fd.RuntimeOption()
op.use_gpu()
op.use_trt_backend()
op.trt_option.serialize_file = 'yolov8s_pt_hand_cache_224'
op1 = fd.RuntimeOption()
op1.use_gpu()
op1.use_trt_backend()
op1.trt_option.serialize_file = 'tinypose_hand_cache'
model_detect = fd.vision.detection.YOLOv8('runs/detect/train2/weights/best.onnx', runtime_option=op)
model_detect.preprocessor.size = [224, 224]

model_keypoints = fd.vision.keypointdetection.PPTinyPose('keypoint_output_inference/tinypose_256x192/model.pdmodel',
                                                         'keypoint_output_inference/tinypose_256x192/model.pdiparams',
                                                         'keypoint_output_inference/tinypose_256x192/infer_cfg.yml',
                                                         runtime_option=op1)
capture = cv2.VideoCapture(0)
cv2.namedWindow('hand', 0)
cv2.resizeWindow('hand', [900, 750])

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.namedWindow('canvas', 0)
cv2.resizeWindow('canvas', [900, 750])

first_point = True
kf = None

while True:
    ret, frame = capture.read()
    if ret:
        frame = cv2.flip(frame, 1)
        result = model_detect.predict(frame)
        if len(result.boxes) > 0:
            xmin, ymin, xmax, ymax = result.boxes[0]
            image_crop = np.array(frame[int(ymin):int(ymax), int(xmin):int(xmax)])
            re_keypoints = model_keypoints.predict(image_crop)
            # vis_im = fd.vision.vis_detection(frame, result, score_threshold=0.65)
            keypoints = re_keypoints.keypoints
            scores = re_keypoints.scores
            final_draw = draw_key_point.vis_keypoints(frame, keypoints, [xmin, ymin], scores)
            if scores[8] > 0.5 and math.sqrt(
                    (keypoints[8][0] - keypoints[4][0]) ** 2 + (keypoints[8][1] - keypoints[4][1]) ** 2) <= 20:
                x, y = keypoints[8][0] + xmin, keypoints[8][1] + ymin
                if first_point:
                    first_point = False
                    kf = kf2d.KalmanFilter2D(np.array([x, y]))
                else:
                    x, y = kf.update(np.array([x, y]))
                cv2.circle(canvas, (int(x), int(y)), 5, color=[0, 0, 255],
                           thickness=-1)
            else:
                first_point = True
                kf = None
            if math.sqrt((keypoints[12][0] - keypoints[4][0]) ** 2 + (keypoints[12][1] - keypoints[4][1]) ** 2) <= 20:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            img = cv2.addWeighted(final_draw, 0.3, canvas, 0.7, 0)
            cv2.imshow('hand', final_draw)
            cv2.imshow('canvas', canvas)
        else:
            cv2.imshow('hand', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv2.destroyWindow('hand')
cv2.destroyWindow('canvas')
