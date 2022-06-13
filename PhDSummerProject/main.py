#Standard packages
import os, sys
import cv2
from pynput.keyboard import Key, Listener, KeyCode
import numpy as np
import matplotlib.pyplot as MPL

#Local files
import init_directories
import capture
import ImageProcessor
from Utilities import remove_block_images, remove_background_images, generate_labels, unravel_FFGEI, create_HOGFFGEI, generate_instance_lengths
import GEI
import LocalResnet
import Experiment_Functions

#Torch
import torch
from torchvision.transforms import ToTensor, Lambda

#MaskCNN only works on the PC version of this app, the Jetson Nano doesn't support python 3.7
if sys.version_info[:3] > (3, 7, 0):
    import maskcnn

#0 is main menu, 1 is second menu, 2 is verbosity selection menu
current_menu = 0
selected_function = None 

def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)
    
def run_camera(path="./Images/Instances/", v=0):
    try:
        camera = capture.Camera()
        camera.run(path="./Images/Instances/", verbose=v)
    except:
        main("No camera detected, returning to main menu")

def get_silhouettes(v =1):
    ImageProcessor.get_silhouettes('./Images/Instances', verbose=v)
    
def on_press(key):
    global current_menu
    global selected_function

    if hasattr(key, 'char'):
        if key.char == '1':
            if current_menu == 0:
                selected_function = run_camera
                verbosity_selection(max_verbose = 2)
            elif current_menu == 1:
                #Load in data for testing
                create_HOGFFGEI()
                main()
            elif current_menu == 2:
                selected_function(v= 0)
                main()
                
        if key.char == '2':
            if current_menu == 0:
                selected_function = get_silhouettes
                verbosity_selection(max_verbose = 2)
            elif current_menu == 1:
                remove_background_images('./Images/Archive/Background_clean_test')
                main()
                print("backgrounds removed sucessfully.")
            elif current_menu == 2:
                selected_function(v=1)
                main()
                
        if key.char == '3':
            if current_menu == 0:
                #GEI.create_standard_GEI('./Images/GraphCut', './Images/GEI/GraphCut/')
                #GEI.create_standard_GEI('./Images/SpecialSilhouettes', './Images/GEI/SpecialSilhouettes/')
                GEI.create_standard_GEI('./Images/Masks', './Images/GEI/Masks/', mask = True)
            elif current_menu == 1:
                if sys.version_info[:3] > (3, 7, 0):
                    print("true")
                    cnn_segmenter = maskcnn.CNN_segmenter()
                    cnn_segmenter.load_images('./Images/Instances')
                    print("images loaded")
                    masks = cnn_segmenter.detect()
                    main()
                    print("CNN-based segmentation complete")
                else:
                    print("Wrong version of python: cannot complete operation, you need python 3.7 or higher.")
            else:
                selected_function = None

            main()
        if key.char == '4':
            if current_menu == 0:
                #Three options for creating FFGEIS
                #GEI.create_FF_GEI('./Images/GraphCut', './Images/FFGEI/GraphCut/')
                #GEI.create_FF_GEI('./Images/SpecialSilhouettes', './Images/FFGEI/SpecialSilhouettes/')
                GEI.create_FF_GEI('./Images/Masks', './Images/FFGEI/Masks/', mask = True)
            elif current_menu == 1:
                batch_size = 3
                epoch = 15
                target = Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
                train_val_loader, test_loader= LocalResnet.create_dataloaders(sourceTransform=ToTensor(),
                                                                        targetTransform = target,
                                                                        labels = './labels/labels.csv',
                                                                        images = './Images/GEI/SpecialSilhouettes',
                                                                        sizes = './Instance_Counts/normal/GEI.csv',
                                                                        batch_size = batch_size,
                                                                        FFGEI = False)
                print("datasets prepared sucessfully")
                model = LocalResnet.train_network(train_val_loader, test_loader, epoch = epoch, batch_size = batch_size, out_path = './Results/FFGEI/GraphCut/', model_path = './Models/FFGEI_GraphCut/')
                #Experiments:
                #GEI experiments:
                #'./Labels/labels.csv,' './Images/GEI/SpecialSilhouettes' #All 42 long, run for 15 epochs, batch size 3 -
                #'./Labels/labels.csv,' './Images/GEI/GraphCut'
                #'./Labels/labels.csv,' './Images/GEI/Masks' -

                #FFGEI standard, needs extra code for going into each individual instance folder # All 4099 long, run for 3 epochs, batch size 50
                #'./labels/FFGEI_labels.csv' './Images/FFGEI/Unravelled/SpecialSilhouettes', './Models/FFGEI_Special/' - done
                #'./labels/FFGEI_labels.csv' './Images/FFGEI/Unravelled/Masks' - done
                #'./labels/FFGEI_graphcut_labels.csv' './Images/FFGEI/Unravelled/GraphCut' -
                
                #FFGEI imbued with HOG, all 4099 long, run for 3 epochs, batch size 50
                #'./labels/FFGEI_labels.csv' './Images/HOGFFGEI/SpecialSilhouettes' -
                #'./labels/FFGEI_labels.csv' './Images/HOGFFGEI/Mask' -
            main()
            print("network training and testing complete")
        if key.char == '5':
            if current_menu == 0:
                ImageProcessor.get_silhouettes('./Images/Instances', HOG=True)
            elif current_menu == 1:
                generate_labels('./Images/FFGEI/Graphcut', out='./Labels/FFGEI_graphcut_labels.csv')
            main()
        if key.char == '6':
            if current_menu == 0:
                ImageProcessor.create_special_silhouettes()
                main()
                print("Special silhouettes created.")
            elif current_menu == 1:
                Experimental_Functions.process_input_video('./Images/Instances/Instance_0.0', './Images/Masks/Instance_0.0')
            else:
                main()
        if key.char == '7':
            if current_menu == 0:
                ImageProcessor.graph_cut()
                main()
                print("Graph cut operation completed.")
            elif current_menu == 1:
                Experiment_Functions.compare_ground_truths('./Images/Ground Truths', ['./Images/SpecialSilhouettes', './Images/Masks', './Images/GraphCut'])
                main()
                print("ground truth comparison completed.")
        if key.char == '8':
            if current_menu == 0:
                extended_menu()
            elif current_menu == 1:
                # normal and graphcut the only differing values as graphcut is the only function to discard frames.
                generate_instance_lengths('./Images/SpecialSilhouettes', './Instance_Counts/normal/' )
                main()
                print("instance indices generated")
        if key.char == '9':
            if current_menu == 0:
                return False
            elif current_menu == 1:
                main()

#Verbosity selection for camera and image processing functions
def verbosity_selection(max_verbose = 1):
    clear_console()
    global current_menu
    current_menu = 2
    print("current menu", current_menu)
    print("Choose Verbosity")
    print("Select one of the following options:\n")
    for i in range(0, max_verbose):
        print(str(i+1) + ". " + str(i))
    print(str(max_verbose + 1) + ". Back")

def extended_menu():
    global current_menu
    current_menu = 1
    clear_console()
    print("More")
    print("Select one of the following options:",
          "\n\nUTILITIES\n",
          "1. Test data and labels\n",
          "2. Remove backgrounds\n",
          "3. CNN segmenter\n",
          "4. Experimental Resnet\n",
          "5. Generate labels\n",
          "6. Run video prediction\n",
          "7. Compare ground truths\n",
          "8. Generate instance lengths\n",
          "9. Back\n")

def main(error_message = None, init = False):
    global current_menu
    current_menu = 0
    clear_console()
    if error_message:
        print(error_message)

    print("Welcome")
    print("Select one of the following options:",
          "\n\nREGULAR FUNCTIONS\n",
          "1. Activate camera capture",
          "\n\nPRE-PROCESSING\n",
          "2. Process background\n", # needs verbosity
          "3. Create standard GEIs\n",
          "4. Create FF-GEIs\n",
          "5. Generate HOG images\n",
          "6. Special Silhouettes\n",
          "7. Graph Cut\n",
          "8. More\n",
          "9. Quit")
    if init == True:
        with Listener(on_press=on_press) as listener:
            try:
                listener.join()
            except:
                print("program ended, listener closing")

if __name__ == '__main__':
    #Main menu
    main(init = True)
