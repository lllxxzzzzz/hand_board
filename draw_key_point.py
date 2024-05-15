import cv2
import copy


def vis_keypoints(img, keypoints, bbox_min=None, scores=None):
    if len(keypoints) == 0:
        return img
    canvas = img.copy()
    key_points = copy.deepcopy(keypoints)
    if bbox_min:
        for i in range(len(key_points)):
            key_points[i][0] += bbox_min[0]
            key_points[i][1] += bbox_min[1]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9),
             (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17),
             (17, 18), (18, 19), (19, 20)]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [255, 0, 0]]
    for i in range(len(key_points)):
        if scores[i] <= 0.5:
            continue
        x = int(key_points[i][0])
        y = int(key_points[i][1])
        cv2.circle(canvas, (x, y), radius=2, color=colors[i], thickness=-1)

    for i in edges:
        if scores[i[0]] <= 0.5 or scores[i[1]] <= 0.5:
            continue
        start_x = int(key_points[i[0]][0])
        start_y = int(key_points[i[0]][1])
        end_x = int(key_points[i[1]][0])
        end_y = int(key_points[i[1]][1])
        cv2.line(canvas, (start_x, start_y), (end_x, end_y), color=colors[i[1]])
    del keypoints
    return canvas
