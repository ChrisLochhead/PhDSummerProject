import cv2
import numpy as np
#from hallwayprograms.human_detector.JetsonYolo_Main.elements.yolo import OBJ_DETECTION
from obj_yolo import OBJ_DETECTION

Object_classes =['person']#['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                #'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                #'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                #'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                #'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                #'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                #'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                #'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                #'hair drier', 'toothbrush' ]

Object_colors = list(np.random.rand(1,3)*255)
Object_detector = OBJ_DETECTION('yolov5s.pt', Object_classes)

def gstreamer_pipeline(
    capture_width=424, #640, #1280,
    capture_height=240, #480, #720,
    display_width=424, #640, #1280,
    display_height=240, #480, #720,
    framerate=15, #30, #60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def get_objs_from_frame(frame, lightweight = False):
    return Object_detector.detect(frame, lightweight)

def plot_obj_bounds(objs, frame):
    i = 0
    seen_human = False
    dimensions = []
    for obj in objs:
        if obj['label'] == "person":
            print("seen human")
            seen_human = True
            label = obj['label']
            score = obj['score']
            [(xmin, ymin), (xmax, ymax)] = obj['bbox']
            dimensions = [xmin, ymin, xmax, ymax]
            color = Object_colors[Object_classes.index(label)]
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1,
                                cv2.LINE_AA)
        i += 1
    return frame, dimensions
