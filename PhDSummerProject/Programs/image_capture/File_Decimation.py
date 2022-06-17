from __future__ import print_function

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
import shutil

#Local files
import JetsonYolo
import ImageProcessor
import GEI
import torch
import LocalResnet
from Utilities import numericalSort

#Torch and SKlearn
from torchvision.transforms import ToTensor, Lambda
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Google drive connection
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from apiclient import errors
from apiclient.http import MediaFileUpload

gauth = GoogleAuth()
drive = GoogleDrive(gauth)


# Only works on the PC version of this app, Jetson doesn't support python 3.7
if sys.version_info[:3] > (3, 7, 0):
    import maskcnn
# For printing matplotlib charts from command prompt
matplotlib.use('TkAgg')

# If modifying these scopes, delete the file token.json.
#SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
def init_google_drive():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)

        # Call the Drive v3 API
        results = service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
            return
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))

        return service

    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')


def get_files_in_folder(service, folder_id):
    """Print files belonging to a folder.

    Args:
    service: Drive API service instance.
    folder_id: ID of the folder to print files from.
    """
    page_token = None
    while True:
        try:
            param = {}
            if page_token:
                param['pageToken'] = page_token
            children = service.children().list(
                folderId=folder_id, **param).execute()

            count = 0
            for child in children.get('items', []):
                count += 1
                #print 'File Id: %s' % child['id']
            print("count : ", count)
            page_token = children.get('nextPageToken')
            if not page_token:
                break
        except errors.HttpError:#, error:
            print("error") #'An error occurred: %s' % error
            break


#Function to check if more than 1 person in any frames, and deleting if so
def check_human_count(images):
    lesser_count = 0
    greater_count = 0
    for image in images:
        objs = JetsonYolo.get_objs_from_frame(np.asarray(image), False)
        if len(objs) > 1:
          greater_count += 1
        elif len(objs) < 1:
            lesser_count += 1
    print("human detected: ", greater_count, lesser_count, len(images))
    if greater_count > (len(images) * 0.5) or lesser_count > (len(images) * 0.5):
        return False
    return True

#Function to check person traverses the width of the image, else dump the files
def check_human_traversal(images):
    #Highest possbile pixel value is 240
    min_x = 1000
    max_x = -1000

    for image in images:
        objs = JetsonYolo.get_objs_from_frame(np.asarray(image), False)
        #box coords takes the format [xmin, ymin, xmax, ymax]
        debug_img, box_coords = JetsonYolo.plot_obj_bounds(objs, np.asarray(image))
        #print("box co-ords: ", box_coords)
        #Co-ordinates will only be present in images with a human present
        if len(box_coords) > 0:
            if box_coords[0] < min_x:
                min_x = box_coords[0]
            if box_coords[0] > max_x:
                max_x = box_coords[0]

    #If the difference in bounding boxes is equal to 50% of the frame width
    print("max and min: ", max_x, min_x, images[0].shape[0])
    if abs(max_x - min_x) > (images[0].shape[0] * 0.5):
        return True

    return False

#Function to send proof-read instances to a g-mail account or google drive
def decimate_and_send(path = './Images/CameraTest'):
    #Instead of deleting, flag the indices to see which ones would get decimated to ascertain if it's acceptable.
    #Load in images
    service = init_google_drive()
    instances = []
    image_names = []
    folder_names = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if iterator == 0:
            folder_names = dirs
            print("folders: ", folder_names, type(folder_names))
        images = []
        im_names = []
        if len(files) > 0:
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                # load the input image and associated mask from disk
                image = cv2.imread(os.path.join(subdir, file))
                #print("file name: ", file)
                images.append(image)
                im_names.append(os.path.join(subdir, file))
            print("len: ", len(images))
            image_names.append(im_names)
            instances.append(images)

    master_folder_id = '11kK5Rrxw5Dp3a_vXYowYNhglHjQmLna8'

    kwargs = {
        "q": "'{}' in parents".format(master_folder_id),
        # Specify what you want in the response as a best practice. This string
        # will only get the files' ids, names, and the ids of any folders that they are in
        #"fields": "nextPageToken,incompleteSearch,files(id,parents,name)",
        # Add any other arguments to pass to list()
    }
    request = service.files().list(**kwargs)
    print(request)
    response = request.execute()
    #print("number of sub folders: ", len(response['files']))
    start_count = len(response['files'])
    # for r in response['files']:
    #    print(r)
    print("start: ", start_count)

    for i, ims in enumerate(instances):
        print("enumerating instance: ", i)
        #Pass images through check human count
        contains_1_human = check_human_count(ims)
        #Pass images through check human traversal
        human_traverses_fully = check_human_traversal(ims)

        #Send images to google drive
        if contains_1_human and human_traverses_fully:
            print("instance : ", i, " passes")
            #Send to google drive
            #11kK5Rrxw5Dp3a_vXYowYNhglHjQmLna8 is the folder ID

            #Create a sub folder for each instance

            #Going to need to work out a way to account for if the folder already has certain instances
            print("starting count is ", start_count)
            file_metadata = {
                'name': 'instance_' + str(float(i + start_count)),
                'parents': [master_folder_id],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, supportsAllDrives=True).execute()
            #print(this_will_fail)

            #folder = drive.CreateFile(file_metadata)
            #folder.Upload()

            folder_id = folder['id']#
            print("folder ID: ", folder_id)#folder.getId()

            for iter, im in enumerate(ims):
                file_metadata = {'name': image_names[i][iter],
                                 'parents': [folder_id]}

                media = MediaFileUpload(image_names[i][iter],
                                        mimetype='image/jpeg')
                file = service.files().create(body=file_metadata,
                                                    media_body=media,
                                                    fields='id').execute()

                # Read file and set it as the content of this instance.
                print("content path: ", path + "/" + image_names[i][iter])
                #gfile.SetContentFile(image_names[i][iter])
                #gfile.Upload()  # Upload the file.

        else:
            print("instance : ", i, " fails: ", contains_1_human, human_traverses_fully)
            #delete folder
            delete_folder(folder_names[i])
            
    print("made it out of the loop?")  
        


def delete_folder(index, path = './Images/CameraTest/'):
    deletion_folder = str(path + str(index))
    #This will delete the folder and its contents recursively
    shutil.rmtree(deletion_folder, ignore_errors=True)