#Standard
import cv2
import re
from PIL import Image, ImageOps
import numpy as np
import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

#Local files
import JetsonYolo
import ImageProcessor
import GEI
import torch
import LocalResnet

#Torch and SKlearn
from torchvision.transforms import ToTensor, Lambda
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Only works on the PC version of this app, Jetson doesn't support python 3.7
if sys.version_info[:3] > (3, 7, 0):
    import maskcnn
#For printing matplotlib charts from command prompt
matplotlib.use('TkAgg')

########## Utility functions ####################
#################################################
def compare_ground_truths(ground_truth_path, raw_image_paths, out_path = './Results/Ground Truths/'):
    #ground truths: instance 5, 6, 25, 26 using frames 1,2,3,4,5,  6,7,8,9,10 and 31,32,33,34,35
    #Load in ground truths to array
    ground_truths = []
    raw_directories = ['Instance_5.0', 'Instance_6.0', 'Instance_25.0', 'Instance_26.0']
    sub_directories = ['./Images/SpecialSilhouettes\\' , './Images/Masks\\' , './Images/GraphCut\\' ]
    raw_indices = [[5,6,7,8,9], [30,31,32,33,34]]
    types = [0,1,2] # corresponds to : ['silhouette', 'mask', 'graphcut']
    error_table = []
    averages_table = []

    #Load in ground truths
    for iterator, (subdir, dirs, files) in enumerate(os.walk(ground_truth_path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                # load the input image and associated mask from disk and perform initial pre-processing
                image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                mask =  align_image(image, 30)
                ground_truths.append(mask)

    for type_iter, raw_image_path in enumerate(raw_image_paths):
        #Load in corresponding raw images into another array
        raw_images = []
        for iterator, (subdir, dirs, files) in enumerate(os.walk(raw_image_path)):
            dirs.sort(key=numericalSort)
            if len(files) > 0:
                for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                    for i, _ in enumerate(raw_directories):
                        if(subdir == sub_directories[type_iter] + raw_directories[i]):
                            # load the input image and associated mask from disk and perform initial pre-processing
                            if file_iter in raw_indices[0] and i == 0 or file_iter in raw_indices[0] and i == 2:
                                #Masks arent saved in aligned silhouettes cause they are used to create special silhouettes
                                if type_iter == 1:
                                    image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                                    mask = align_image(image, 30)
                                    raw_images.append(mask)
                                else:
                                    raw_images.append(cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE))

                            elif file_iter in raw_indices[1] and i == 1 or file_iter in raw_indices[1] and i == 3:
                                if type_iter == 1:
                                    image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                                    mask = align_image(image, 30)
                                    raw_images.append(mask)
                                else:
                                    raw_images.append(cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE))

        #print("final lens: ", len(ground_truths), len(raw_images))

        error_rates = []
        #Iterate through ground truths and compare them using multiple different measurement metrics
        for i, ground_truth in enumerate(ground_truths):
            #Debug
            #cv2.imshow("ground truth ", ground_truth)
            #cv2.imshow("silhouette ", raw_images[i])
            #cv2.waitKey(0)

            #Need function to draw largest rectangle over white pixels in images then perform IOU
            # Return confidence and the following four metrics: DICE score, Intersection over Union, normalized cross-correlation (minus normalisation), signal to noise compression
            dice = dice_coef(ground_truth, raw_images[i]):
            error = mean_squared_error(ground_truth, raw_images[i])
            diff = cv2.absdiff(ground_truth, raw_images[i])
            m_norm = sum(abs(diff))  # Manhattan norm
            z_norm = norm(diff.ravel(), 0)  # Zero norm
            (score, diff) = compare_ssim(ground_truth, raw_images[i], full=True)
            error_rates.append([types[type_iter], error, m_norm, z_norm, score])

        error_rates = np.array(error_rates).astype(float)
        element_average = [types[type_iter], sum(error_rates[:, 1])/len(error_rates[:, 1]), sum(error_rates[:, 2])/len(error_rates[:, 2]),
                           sum(error_rates[:, 3])/len(error_rates[:, 3]), sum(error_rates[:, 4])/len(error_rates[:, 4])]
        element_average = np.array(element_average).astype(float)
        error_rates = np.vstack([error_rates, element_average])

        if len(error_table) <= 0:
            error_table = error_rates
        else:
            error_table = np.vstack([error_table, error_rates])
    #Record accuracy results + print + save to a .csv
    os.makedirs(out_path, exist_ok=True)
    np.savetxt( out_path + 'error_margins.csv', error_table, fmt='%f', delimiter=",")
    
def process_input_video(instance_path, mask_path, model_path = './Models/FFGEI_Special/model_fold_2.pth', silhouette_type='Special', label_class = 1):
    #Get raw images
    input_frames = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(instance_path)):
        if len(files) > 0:
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                input_frames.append(cv2.imread(os.path.join(subdir, file), 0))
    #Get corresponding masks
    masks = []
    if mask_path != 'none':
        for iterator, (subdir, dirs, files) in enumerate(os.walk(mask_path)):
            if len(files) > 0:
                for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                    masks.append(cv2.imread(os.path.join(subdir, file), 0))
    else:
        # All three versions require a masking algorithm, only using the neural network version for brevity
        # This version only for novel images with no ready-made masks, as making masks takes a while
        # Create Masks
        cnn_segmenter = maskcnn.CNN_segmenter()
        cnn_segmenter.load_images(instance_path)
        masks = cnn_segmenter.detect()

    #Create the silhouettes
    silhouettes = []
    if silhouette_type == 'Special':
        silhouettes = ImageProcessor.create_special_silhouettes(mask_path='none', image_path=instance_path, masks=masks, single = True)
    elif silhouette_type == 'Graph':
        silhouettes = ImageProcessor.graph_cut(mask_path='none', image_path=instance_path, by_mask=True, mask_edges=True, masks=masks)#, single = True)
    elif silhouette_type == 'Mask':
        silhouettes = masks

    #Turn Silhouettes into GEIs
    FFGEIS =  []
    if silhouette_type != 'Mask':
        FFGEIS = GEI.create_FF_GEI('none', 'none', mask=False, single = True, sil_array = silhouettes)
    else:
        FFGEIS = GEI.create_FF_GEI('none', 'none', mask=True, single = True, sil_array = silhouettes)

    # Load neural network model
    network = LocalResnet.ResNet50(img_channel=1, num_classes=2)
    network.load_state_dict(torch.load(model_path))
    network.eval()

    #Create labels
    label_data =[['ID', 'Class']]
    for i in range(len(FFGEIS)):
        label_data.append([i, label_class])
    os.makedirs('./Temp/Labels', exist_ok=True)
    np.savetxt('./Temp/Labels.csv', label_data, delimiter=",", fmt='%s')

    #Save images
    os.makedirs('./Temp/Images', exist_ok=True)
    for iter, im in enumerate(FFGEIS):
        cv2.imwrite('./Temp/Images/' + str(iter) + ".jpg", im)

    target = Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    dataset = LocalResnet.CustomDataset('./Temp/Labels.csv', './Temp/Images',
                            sourceTransform=ToTensor(), targetTransform = target,
                            FFGEI = True)

    #Transform data into a data loader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # [predictions, total_accuracy, total_confidence, precision, recall, f1_score]
    accuracy_results = LocalResnet.check_accuracy(test_loader, network, debug=True)
    # Do a per-frame estimation, tallying up predictions
    for iter in range(len(input_frames)):
        # Display each frame as you go, write the prediction on the screen per frame with the confidence.
        image = input_frames[iter]
        text = 'Claire'
        if accuracy_results[0][iter] == 0:
            text = 'Chris'
        #Annotate image with prediction
        image = cv2.putText(image, 'Class: ' + text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255), 1,
                            cv2.LINE_AA)
        cv2.imshow("Result", image)
        cv2.waitKey(0)
    #Calculate prediction and confidence
    accum_pred = most_frequent(accuracy_results[0])
    claire_count = sum(accuracy_results[0])
    chris_count = len(accuracy_results[0]) - claire_count
    prediction = 'Claire'
    confidence_score = (claire_count / len(accuracy_results[0])) * 100
    if chris_count > len(accuracy_results[0]) / 2:
        prediction = 'Chris'
        confidence_score = (chris_count / len(accuracy_results[0])) * 100
    #Final decision
    print("The individual in this sequence is : ", prediction, " with ", confidence_score, "% of the voted frames.")

#################################################
#################################################