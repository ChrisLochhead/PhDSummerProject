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
import File_Decimation
import Ensemble

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
        camera.run(path="./Images/CameraTest/", verbose=v)
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
                #main()
            elif current_menu == 3:
                # normal and graphcut the only differing values as graphcut is the only function to discard frames.
                generate_instance_lengths('./Images/SpecialSilhouettes', './Instance_Counts/Normal/', ignore_first = True )
                generate_instance_lengths('./Images/GraphCut', './Instance_Counts/Graphcut/', ignore_first = True  )

                #Few Shot
                generate_instance_lengths('./Images/SpecialSilhouettes/FewShot', './Instance_Counts/FewShot/Normal/' )
                generate_instance_lengths('./Images/GraphCut/FewShot', './Instance_Counts/FewShot/GraphCut/' )
                main()
                print("instance indices generated")

        if key.char == '2':
            if current_menu == 0:
                selected_function = get_silhouettes
                verbosity_selection(max_verbose = 2)
            elif current_menu == 1:
                remove_background_images('./Images/Instances')
                main()
                print("backgrounds removed sucessfully.")
            elif current_menu == 2:
                selected_function(v=1)
                main()
            elif current_menu == 3:
                #Conduct file decimation and upload test
                File_Decimation.decimate_and_send()
                main()

        if key.char == '3':
            if current_menu == 0:
                #Standard database
                GEI.create_standard_GEI('./Images/GraphCut', './Images/GEI/GraphCut/')
                GEI.create_standard_GEI('./Images/SpecialSilhouettes', './Images/GEI/SpecialSilhouettes/')
                GEI.create_standard_GEI('./Images/Masks', './Images/GEI/Masks/', mask = True)
                #FewShot data
                GEI.create_standard_GEI('./Images/GraphCut/FewShot', './Images/GEI/FewShot/GraphCut/')
                GEI.create_standard_GEI('./Images/SpecialSilhouettes/FewShot', './Images/GEI/FewShot/SpecialSilhouettes/')
                GEI.create_standard_GEI('./Images/Masks/FewShot', './Images/GEI/FewShot/Masks/', mask = True)
                main()
            elif current_menu == 1:
                if sys.version_info[:3] > (3, 7, 0):
                    print("true")
                    cnn_segmenter = maskcnn.CNN_segmenter()
                    cnn_segmenter.load_images('./Images/FewShot')#Instances')
                    print("images loaded")
                    masks = cnn_segmenter.detect()
                    main()
                    print("CNN-based segmentation complete")
                else:
                    print("Wrong version of python: cannot complete operation, you need python 3.7 or higher.")
            elif current_menu == 2:
                selected_function = None
            elif current_menu == 3:
                main()
        if key.char == '4':
            if current_menu == 0:
                #Three options for creating FFGEIS
                print("stage 1")
                GEI.create_FF_GEI('./Images/GraphCut/FewShot', './Images/FFGEI/GraphCut/FewShot/')
                print("stage 2")
                GEI.create_FF_GEI('./Images/SpecialSilhouettes/FewShot', './Images/FFGEI/SpecialSilhouettes/FewShot/')
                print("stage 3")
                GEI.create_FF_GEI('./Images/Masks/FewShot', './Images/FFGEI/Masks/FewShot/', mask = True)
                #main()
            elif current_menu == 1:
                batch_size = 50
                epoch = 3
                target = Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
                train_val_loader, test_loader= LocalResnet.create_dataloaders(sourceTransform=ToTensor(),
                                                                        targetTransform = target,
                                                                        labels = './labels/FFGEI_labels.csv',
                                                                        images = './Images/HOGFFGEI/SpecialSilhouettes',
                                                                        sizes = './Instance_Counts/Normal/indices.csv',
                                                                        batch_size = batch_size,
                                                                        FFGEI = False)
                print("datasets prepared sucessfully")
                model = LocalResnet.train_network(train_val_loader, test_loader, epoch = epoch, batch_size = batch_size, out_path = './Results/HOGFFGEI/SpecialSilhouettes/Test/', model_path = './Models/HOGFFGEI_SpecialSilhouettes/Test/')
                main()
                #Experiments:, do once with normal then once with few-shot 
                
                #GEI experiments:
                #'./Labels/labels.csv,' './Images/GEI/SpecialSilhouettes' #All 42 long, run for 15 epochs, batch size 3 - done
                #'./Labels/labels.csv,' './Images/GEI/GraphCut' - done
                #'./Labels/labels.csv,' './Images/GEI/Masks' - done

                #FFGEI standard, needs extra code for going into each individual instance folder # All 4099 long, run for 3 epochs, batch size 50
                #'./labels/FFGEI_labels.csv' './Images/FFGEI/Unravelled/SpecialSilhouettes', './Models/FFGEI_Special/' - done
                #'./labels/FFGEI_labels.csv' './Images/FFGEI/Unravelled/Masks' - done
                #'./labels/FFGEI_graphcut_labels.csv' './Images/FFGEI/Unravelled/GraphCut' - done
                
                #FFGEI imbued with HOG, all 4099 long, run for 3 epochs, batch size 50
                #'./labels/FFGEI_labels.csv' './Images/HOGFFGEI/SpecialSilhouettes' - done
                #'./labels/FFGEI_labels.csv' './Images/HOGFFGEI/Mask' - done
            elif current_menu == 3:
                print("creating HOGFFGEI")
                #create_HOGFFGEI(FFGEI_path='./Images/FFGEI/Unravelled/Masks', HOG_path='./Images/FFGEI/Unravelled/HOG_silhouettes',
                #                label='./Labels/FFGEI_labels.csv', out='./Images/HOGFFGEI/Mask/')
                
                #create_HOGFFGEI(FFGEI_path='./Images/FFGEI/Unravelled/SpecialSilhouettes', HOG_path='./Images/FFGEI/Unravelled/HOG_silhouettes',
                #                label='./Labels/FFGEI_labels.csv', out='./Images/HOGFFGEI/SpecialSilhouettes/')

                #Create HOGFFGEI: Few-shot
                #\Images\HOG_silhouettes\FewShot\Unravelled
                create_HOGFFGEI(FFGEI_path='./Images/FFGEI/FewShot/Unravelled/Masks', HOG_path='./Images/HOG_silhouettes/FewShot/Unravelled',
                                label='./Labels/FFGEI_labels.csv', out='./Images/HOGFFGEI/FewShot/Masks/')

                create_HOGFFGEI(FFGEI_path='./Images/FFGEI/FewShot/Unravelled/SpecialSilhouettes', HOG_path='./Images/HOG_silhouettes/FewShot/Unravelled',
                                label='./Labels/FFGEI_labels.csv', out='./Images/HOGFFGEI/FewShot/SpecialSilhouettes/')
                main()
            #print("network training and testing complete")
        if key.char == '5':
            if current_menu == 0:
                #ImageProcessor.get_silhouettes('./Images/FewShot', HOG=True, few_shot = True)
                ImageProcessor.get_silhouettes('./Images/Instances', HOG=True)
                main()
            elif current_menu == 1:
                #Treat the cutoff as if the iterator starts at 1 because folder 0 is always empty.
                generate_labels('./Images/FFGEI/Graphcut', out='./Labels/', name = 'FFGEI_graphcut_labels.csv', cutoff = 20, cutoff_index = 1)
                generate_labels('./Images/FFGEI/SpecialSilhouettes', out='./Labels/', name = 'FFGEI_labels.csv', cutoff = 20, cutoff_index = 1)
                generate_labels('./Images/GEI/Masks', out='./Labels/', name = 'labels.csv', cutoff = 20, cutoff_index = 1)
                main()
            elif current_menu == 3:
                unravel_FFGEI(path='./Images/FFGEI/Unravelled/Masks')
                unravel_FFGEI(path='./Images/FFGEI/Unravelled/GraphCut')
                unravel_FFGEI(path='./Images/FFGEI/Unravelled/SpecialSilhouettes')
                unravel_FFGEI(path='./Images/FFGEI/Unravelled/HOG_silhouettes')

                #unravel FFGEIs
                #unravel_FFGEI(path='./Images/FFGEI/Unravelled/GraphCut')
                #unravel_FFGEI(path='./Images/FFGEI/Unravelled/SpecialSilhouettes')
                #unravel_FFGEI(path='./Images/FFGEI/Unravelled/HOG_silhouettes')

                #FFGEIs : few-shot
                #unravel_FFGEI(path='./Images/FFGEI/FewShot/Unravelled/GraphCut')
                #unravel_FFGEI(path='./Images/FFGEI/FewShot/Unravelled/SpecialSilhouettes')
                #unravel_FFGEI(path='./Images/FFGEI/FewShot/Unravelled/Masks')

                #HOG-FFGEIs : few-shot
                #unravel_FFGEI(path='./Images/FFGEI/FewShot/GraphCut/Unravelled')
                #unravel_FFGEI(path='./Images/FFGEI/FewShot/SpecialSilhouettes/Unravelled')
                #unravel_FFGEI(path='./Images/HOG_silhouettes/FewShot/Unravelled')
                main()
        if key.char == '6':
            if current_menu == 0:
                ImageProcessor.create_special_silhouettes()
                main()
                print("Special silhouettes created.")
            elif current_menu == 1:
                Experimental_Functions.process_input_video('./Images/Instances/Instance_0.0', './Images/Masks/Instance_0.0')
            elif current_menu == 3:
                print("test for menu")
                main()
        if key.char == '7':
            if current_menu == 0:
                ImageProcessor.graph_cut()
                main()
                print("Graph cut operation completed.")
            elif current_menu == 1:
                #Ground truth comparison for the regular dataset
                Experiment_Functions.compare_ground_truths('./Images/Ground Truths/Regular', ['./Images/SpecialSilhouettes', './Images/Masks', './Images/GraphCut'], 
                                                           ['Instance_5.0', 'Instance_25.0'],
                                                           out_path = './Results/Ground Truths/Normal/')
                #Ground truth comparison for the fewshot dataset
                Experiment_Functions.compare_ground_truths('./Images/Ground Truths/FewShot', ['./Images/SpecialSilhouettes/FewShot', './Images/Masks/FewShot', './Images/GraphCut/FewShot'],
                                                           ['Instance_1.0', 'Instance_4.0'],
                                                           out_path = './Results/Ground Truths/FewShot/', few_shot = True)
                main()
                print("ground truth comparison completed.")
            elif current_menu == 3:
                #GEI experiments:
                #SpecialSilhouette
                Ensemble.few_shot_ensemble_experiment(3, batch_size = 3, epoch = 15)
                main()
        if key.char == '8':
            if current_menu == 0:
                extended_menu(1, page_1)
            elif current_menu == 1:
                extended_menu(3, page_3)
        if key.char == '9':
            if current_menu == 1:
                main()
            else:
                return False

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

def extended_menu(index, content):
    global current_menu
    current_menu = index
    clear_console()
    print("More")
    print(content)


page_0 = """Welcome

Select one of the following options:
         
REGULAR FUNCTIONS
         
1. Activate camera capture

PRE-PROCESSING

2. Process background
3. Create standard GEIs
4. Create FF-GEIs
5. Generate HOG images
6. Special Silhouettes
7. Graph Cut
8. More
9. Quit"""

page_1 = """Select one of the following options:

UTILITIES

1. Test data and labels
2. Remove backgrounds
3. CNN segmenter
4. Experimental Resnet
5. Generate labels
6. Run video prediction
7. Compare ground truths
8. More
9. Back\n"""

page_3 = """Select on of the following options:

1. Generate instance lengths
2. Test file decimation and sending system
3. Back
4. Create HOGFFGEIs
5. Unravel FFGEIs
6. menu debug
7. few-shot ensemble learning"""

def main(error_message = None, init = False):
    global current_menu
    current_menu = 0
    #clear_console()
    if error_message:
        print(error_message)

    print(page_0)

    if init == True:
        with Listener(on_press=on_press) as listener:
            try:
                listener.join()
            except:
                print("program ended, listener closing")

if __name__ == '__main__':
    #Main menu
    main(init = True)
