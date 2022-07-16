import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
#from Utilities import numericalSort, make_directory
import Utilities
# Root directory of the project
#C:\Users\Chris\Desktop\PhDProject\PhDSummerProject\Programs\mask_network\Mask_RCNN
ROOT_DIR = os.path.abspath("../") + "/PhDSummerProject/Programs/mask_network/Mask_RCNN"
print("root: ", ROOT_DIR)
import warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

########## Silhouette functions #################
#################################################
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class CNN_segmenter():
    def __init__(self):
        # Go back to first area:
        os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))
        # Initialise
        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(ROOT_DIR, "logs")  ##
        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")  #
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)  #
        self.config = InferenceConfig()
        self.config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=self.config)
        # Load weights trained on MS-COCO
        self.model.load_weights('mask_rcnn_coco.h5', by_name=True)
        # COCO Class names
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

    def load_images(self, dir):
        self.image_instances = []
        #Go back to where images are stored
        os.chdir("../..")
        print("os ", os.getcwd())
        for iterator, (subdir, dirs, files) in enumerate(os.walk(dir)):
            dirs.sort(key=Utilities.numericalSort)
            if len(files) > 0:
                raw_images = []
                for file_iter, file in enumerate(sorted(files, key=Utilities.numericalSort)):
                    has_human = False
                    image = cv2.imread(os.path.join(subdir, file))
                    raw_images.append(image)
                self.image_instances.append(raw_images)
        print("images loaded to segmenter")

    def detect(self):
        # Run detection
        self.mask_image_instances = []
        #os.chdir("../..")
        print("os in detect ", os.getcwd())

        for iter, images in enumerate(self.image_instances):
            human_masks = []
            for i, image in enumerate(images):
                print("processing image: ", i, " out of ", len(images), " in instance ", iter, ".")
                human_present = False
                results = self.model.detect([image], verbose=0)
                r = results[0]
                for id in r['class_ids']:
                    if id == 1:
                        human_present = True
                        print("human detected")
                        print(self.class_names[id]) # person is ID of 1

                mask = r['masks']
                mask = mask.astype(int)
                mask.shape

                #If no humans are present, just send an empty mask through
                empty_mask = np.zeros((mask.shape[0],mask.shape[1]))
                if human_present == False:
                    human_masks.append(empty_mask.astype(np.uint8))
                    continue

                #For every colour dimension
                for i in range(mask.shape[2]):
                    if r['class_ids'][i] == 1:
                        #Get mask of the human, change from 0 - 1 to 0-255 for visualisation and debugging before
                        #Adding it to the save array
                        person_mask = mask[:, :, i]
                        person_mask[person_mask == 1] = 255

                human_masks.append(person_mask.astype(np.uint8))
            self.mask_image_instances.append(human_masks)

        #Make mask directory if it doesn't exist yet
        try:
            os.mkdir(os.getcwd() + "\Images\Masks")
        except:
            print("tried to make this: ", os.getcwd() + "\Images\Masks")

        for instance in self.mask_image_instances:
            #Make directory for each instance
            directory_made = False
            n = 0.0
            while directory_made == False:
                try:
                    os.mkdir(os.getcwd() + "\Images\Masks\Instance_" + str(n))
                    directory_made = True
                except:
                    print("already exists: ", n)
                    n += 1
            for i, image in enumerate(instance):
                cv2.imwrite(os.getcwd() + "\Images\Masks\Instance_" + str(n) + "/" + str(i) + ".jpg", image)

        return self.mask_image_instances
