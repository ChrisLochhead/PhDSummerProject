import sys, os

inletPath = os.path.dirname(__file__)
print("Initialising Packages")
sys.path.append(inletPath + "/Programs/human_detector/JetsonYolo_Main")
sys.path.append(inletPath + "/Programs/human_detector/JetsonYolo_Main/elements")
sys.path.append(inletPath + "/Programs/human_detector/JetsonYolo_Main/models")
sys.path.append(inletPath + "/Programs/human_detector/JetsonYolo_Main/utils")
sys.path.append(inletPath + "/Programs/image_capture")
sys.path.append(inletPath + "/Programs/image_processing")
sys.path.append(inletPath + "/Programs/mask_network/Mask_RCNN/samples")
sys.path.append(inletPath + "/Programs/Resnet")