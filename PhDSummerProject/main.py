import init_directories
import capture
import ImageProcessor
from Utilities import remove_block_images
import os, sys
import cv2
from pynput.keyboard import Key, Listener, KeyCode
import numpy as np
import matplotlib.pyplot as MPL
import GEI

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
    #try:
    camera = capture.Camera()
    camera.run(path="./Images/Instances/", verbose=v)
    #except:
    #    main("No camera detected, returning to main menu")

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
            elif current_menu == 2:
                selected_function(v= 0)
                main()
                
        if key.char == '2':
            if current_menu == 0:
                selected_function = get_silhouettes
                verbosity_selection(max_verbose = 2)
            elif current_menu == 2:
                selected_function(v=1)
                #main()
                
        if key.char == '3':
            if current_menu == 0:
                #GEI.create_standard_GEI('./Images/GraphCut', './Images/GEI/GraphCut/')
                #GEI.create_standard_GEI('./Images/SpecialSilhouettes', './Images/GEI/SpecialSilhouettes/')
                GEI.create_standard_GEI('./Images/Masks', './Images/GEI/Masks/', mask = True)
            else:
                selected_function = None

            main()
        if key.char == '4':
            #GEI.create_FF_GEI('./Images/GraphCut', './Images/FFGEI/GraphCut/')
            #GEI.create_FF_GEI('./Images/SpecialSilhouettes', './Images/FFGEI/SpecialSilhouettes/')
            GEI.create_FF_GEI('./Images/Masks', './Images/FFGEI/Masks/', mask = True)

            main()
        if key.char == '5':
            ImageProcessor.get_silhouettes('./Images/Instances', HOG=True)
            main()
        if key.char == '6':
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
        if key.char == '7':
            ImageProcessor.create_special_silhouettes()
            main()
            print("Special silhouettes created.")
        if key.char == '8':
            ImageProcessor.graph_cut()
            main()
            print("Graph cut operation completed.")
        if key.char == '9':
            return False

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
          "6. CNN segmentation\n",
          "7. Make Special Silhouettes\n",
          "8. Graph cut\n",
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
