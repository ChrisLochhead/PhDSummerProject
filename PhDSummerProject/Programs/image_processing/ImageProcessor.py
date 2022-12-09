#Standard
import numpy as np
import cv2
import os 
import copy
from PIL import Image, ImageFilter
import time

#Local files
from Utilities import make_directory, align_image, get_from_directory, save_to_directory, numericalSort
from HOG_functions import process_HOG_image, get_HOG_image
import JetsonYolo

#SCIPY and SKlearn
from scipy.signal import savgol_filter, fftconvolve
from sklearn.metrics import mean_squared_error
from scipy.linalg import norm
from scipy import sum, average
from skimage.metrics import structural_similarity as compare_ssim


def run_histogram_equalization(image):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img

def edge_detect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255;
    return sobel
   
def create_special_silhouettes(mask_path = './Images/Masks/FewShot', image_path = './Images/FewShot', masks = None, single = False):
    #Allow for the passing of pre-loaded silhouettes for the video test function
    if masks == None:
        mask_instances = get_from_directory(mask_path)
    else:
        mask_instances = masks
        
    special_silhouettes = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for iterator, (subdir, dirs, files) in enumerate(os.walk(image_path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            masks = []
            background = []
            combined_example = []
            silhouettes = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                # load the input image and associated mask from disk and perform initial pre-processing
                image = cv2.imread(os.path.join(subdir, file))
                hist_image = run_histogram_equalization(image)
                blurred = cv2.GaussianBlur(hist_image, (3, 3), 0)
                cv2.imshow("Stage 1", blurred)

                #Prepare the image using a sobel edge detector, remove noise and convert to an 8-bit array
                edgeImg = np.max(np.array([edge_detect(blurred[:, :, 0]), edge_detect(blurred[:, :, 1]), edge_detect(blurred[:, :, 2])]), axis=0)
                mean = np.mean(edgeImg);
                edgeImg[edgeImg <= mean] = 0;
                edgeImg_8u = np.asarray(edgeImg, np.uint8)

                #If first frame in the sequence, set as background
                if file_iter == 0:
                    background = edgeImg_8u

                #Use morphological operations to produce an inflated silhouette from background subtraction
                background_based_silhouette = cv2.absdiff(edgeImg_8u, background)
                cv2.imshow("Stage 2", background_based_silhouette)

                background_based_silhouette = cv2.threshold(background_based_silhouette, 100, 255, cv2.THRESH_BINARY)[1]
                *_, bk_mask = cv2.threshold(background_based_silhouette, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                opening = cv2.morphologyEx(bk_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                cv2.imshow("Stage 3", opening)
                # Retrieve and dilate the mask to prevent parts of the body being excluded
                if single == False:
                    mask = mask_instances[iterator - 1][file_iter]
                else:
                    mask = mask_instances[file_iter]

                bk_expanded = cv2.dilate(mask, kernel, iterations=5)
                cv2.imshow("Stage 4", bk_expanded)
                bk_mask_test = cv2.bitwise_and(opening, opening, mask=bk_expanded)
                cv2.imshow("Stage 5", bk_mask_test)
                #Perform morphological operations on edge detected image after applying mask
                #Thresholding to produce a silhouette of the extremities of the silhouette that the mask may have missed
                edge_based_silhouette = edgeImg_8u * bk_expanded
                edge_based_silhouette = cv2.morphologyEx(edge_based_silhouette, cv2.MORPH_CLOSE, kernel)
                edge_based_silhouette = cv2.morphologyEx(edge_based_silhouette, cv2.MORPH_OPEN, kernel)

                #Get copy of the mask and turn into an actual mask (from range 0-1)
                temp_mask = copy.deepcopy(mask)
                temp_mask[temp_mask == 255] = 1

                #Remove noise from the blurred image as a template
                threshold_lower = 30
                threshold_upper = 220
                mask_based_silhouette = cv2.Canny(blurred, threshold_lower, threshold_upper)
                #Apply the mask
                mask_based_silhouette = cv2.bitwise_and(mask_based_silhouette, mask_based_silhouette, mask=temp_mask)

                #Take this and turn all pixels white, then perform open and close to tidy it up
                mask_based_silhouette = cv2.morphologyEx(mask_based_silhouette, cv2.MORPH_CLOSE, kernel)
                mask_based_silhouette = cv2.morphologyEx(mask_based_silhouette, cv2.MORPH_OPEN, kernel)

                #Align images
                mask =  align_image(mask, 30)
                bk_mask_test =  align_image(bk_mask_test, 1)
                mask_based_silhouette =  align_image(mask_based_silhouette, 30)
                edge_based_silhouette =  align_image(edge_based_silhouette, 30)

                alpha = 1.0
                beta = 1.0
                combined_example = []
                finished_example = []
                if len(combined_example) == 0:
                    combined_example = mask

                #Combine the masks, apply a mode filter to smooth the result
                combined_example = cv2.addWeighted(bk_mask_test, alpha, combined_example, beta, 0.0)
                cv2.imshow("Stage 5", combined_example)
                image_example =  Image.fromarray(combined_example)
                image_example = image_example.filter(ImageFilter.ModeFilter(size=5))
                cv2.imshow("Stage 6", np.asarray(image_example))
                key = cv2.waitKey(0) & 0xff
                silhouettes.append(np.array(image_example))
            special_silhouettes.append(silhouettes)
            print("silhouette complete")
            #If a single image has been passed instead of a whole instance, return after one iteration
            if single == True:
                return silhouettes
    #Save
    save_to_directory(special_silhouettes, './Images/SpecialSilhouettes/FewShot')
    print("operation complete, special silhouettes saved")
    return special_silhouettes

#Graph cut
def graph_cut(mask_path = './Images/Masks/FewShot', image_path = './Images/FewShot', by_mask = True, mask_edges = True, masks = None):

    #Adjust save path depending on which combination is used to create it. The best reults are hard-coded into the definition
    if by_mask and mask_edges:
        save_path = './Images/GraphCut/FewShot'
    elif by_mask and not mask_edges:
        save_path = './Images/graph_mask_noedges'
    elif not by_mask and mask_edges:
        save_path = './Images/graph_nomask_edges'
    else:
        save_path = './Images/graph_nomask_noedges'

    #Allow masks to be read directly from memory for live testing
    if masks == None:
        mask_instances = get_from_directory(mask_path)
    else:
        mask_instances = masks
        
    image_instances = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(image_path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            masks = []
            images = []
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                # load the input image and associated mask from disk
                image = cv2.imread(os.path.join(subdir, file))
                #Emphasize outlines
                image = run_histogram_equalization(image)
                #Blur to remove noise
                blurred = cv2.GaussianBlur(image, (5, 5), 0)
                #Generate edge detection image
                edgeImg = np.max(np.array([edge_detect(blurred[:, :, 0]), edge_detect(blurred[:, :, 1]), edge_detect(blurred[:, :, 2])]), axis=0)
                mean = np.mean(edgeImg);
                # Reduce noise
                edgeImg[edgeImg <= mean] = 0;
                edgeImg_8u = np.asarray(edgeImg, np.uint8)

                rect = [(0, 1), (0, 1)]
                #Bounding box
                if by_mask == False:
                    # Get humans
                    objs = JetsonYolo.get_objs_from_frame(np.asarray(image), False)
                    seen_human = False
                    for obj in objs:
                        (xmin, ymin), (xmax, ymax) = obj['bbox']
                        rect = [xmin, ymin, xmax, ymax]
                    # Detector only returns human objs
                    if len(objs) == 0:
                        continue
                else:
                    # Mask
                    mask = mask_instances[iterator-1][file_iter]
                    if np.all(mask == 0):
                        continue
                    # any mask values greater than zero should be set to probable
                    # foreground
                    mask[mask > 0] = cv2.GC_FGD
                    mask[mask == 0] = cv2.GC_BGD

                # allocate memory for two arrays that the GrabCut algorithm internally
                # uses when segmenting the foreground from the background
                fgModel = np.zeros((1, 65), dtype="float")
                bgModel = np.zeros((1, 65), dtype="float")

                # apply GrabCut using the the mask segmentation method
                start = time.time()

                if mask_edges == True:
                    edgeImg = edgeImg.astype("uint8")
                    edgeImg[edgeImg == 255] = 1;
                    grab_image = cv2.bitwise_and(image, image, mask=edgeImg)
                else:
                    grab_image = image

                if by_mask == False:
                    mask = np.zeros(image.shape[:2], dtype="uint8")

                    (mask, bgModel, fgModel) = cv2.grabCut(grab_image, mask, rect, bgModel,
                                                           fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
                else:
                    (mask, bgModel, fgModel) = cv2.grabCut(grab_image, mask, None, bgModel,
                                                           fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

                end = time.time()
                print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))


                outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                                      0, 1)
                # scale the mask from the range [0, 1] to [0, 255]
                outputMask = (outputMask * 255).astype("uint8")
                #Smooth to avoid noisy pixels on the mask edges
                outputMask = Image.fromarray(outputMask)
                outputMask = outputMask.filter(ImageFilter.ModeFilter(size=13))
                outputMask = np.array(outputMask)
                # apply a bitwise AND to the image using our mask generated by GrabCut to generate our final output image
                output = cv2.bitwise_and(image, image, mask=outputMask)
                outputMask = align_image(outputMask, 0)
                images.append(outputMask)
            image_instances.append(images)
    save_to_directory(image_instances, save_path)
    print("graph cut operation complete")
    
def process_image(image, raw_img, verbose = 0, subtractor = None):
    # Extract the contours formed by the silhouette, image is now silhouette and raw image not used, nor is subtractor
    white_mask = cv2.inRange(image, 180, 255)
    return np.asarray(align_image(white_mask, 0)) # was image

def get_silhouettes(path, verbose = 0, HOG = False, few_shot = False):
    global HOG_background
    mask_instances = get_from_directory('./Images/Masks')
    make_directory(path, "Silhouette folder already exists")
    processed_images = []

    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            raw_images = []
            processed_instances = []
            subtractor = cv2.createBackgroundSubtractorKNN()
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                raw_images.append(cv2.imread(os.path.join(subdir, file)))
                #Prepare image
                gray_img = cv2.cvtColor(raw_images[file_iter], cv2.COLOR_BGR2GRAY)
                #First pass: if HOG take a background example
                if file_iter == 0 and HOG == True:
                    HOG_background = get_HOG_image(gray_img)
                #Process image according to chosen processing method
                if HOG == False:
                    processed_instances.append(process_image(gray_img, raw_images[file_iter], verbose, subtractor))
                else:
                    processed_instances.append(process_HOG_image(get_HOG_image(gray_img), HOG_background, mask_instances[iterator-1][file_iter]))

            processed_images.append(processed_instances)
    # Processed images taken, save to location
    os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))

    for instance in processed_images:
        #Find the latest un-made path and save the new images to it
        path_created = False
        n = 0.0
        while path_created == False:
            try:
                if HOG == False and few_shot == False:
                    local_path = './Images/Silhouettes' + "/Instance_" + str(n) + "/"
                elif HOG == True and few_shot == False:
                    local_path = './Images/HOG_silhouettes' + "/Instance_" + str(n) + "/"
                elif HOG == False and few_shot == True:
                    local_path = './Images/HOG_silhouettes/FewShot' + "/Instance_" + str(n) + "/"
                elif HOG == True and few_shot == True:
                    local_path = './Images/HOG_silhouettes/FewShot' + "/Instance_" + str(n) + "/"
                os.makedirs(local_path, exist_ok=False)
                path_created = True
            except:
                n += 1
        for i, image in enumerate(instance):
            #Exclude entirely black or entirely white images from the sequence.
            if HOG:
                cv2.imwrite(local_path + str(i) + ".jpg", image)
            else:
                if not np.all((image == 0)) and not np.all((image == 255)):
                    cv2.imwrite(local_path + str(i) + ".jpg", image)
    print("all saved")


