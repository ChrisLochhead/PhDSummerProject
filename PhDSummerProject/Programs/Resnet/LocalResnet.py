#Torch
import torch
import torchvision
from torch import optim 
from torch import nn  
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets 
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda

#Standard imports
from tqdm import tqdm  
import pandas as pd
import copy
from numpy.random import default_rng
import matplotlib.pyplot as plt
from PIL import ImageTk, Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
import cv2
import random

#Local files
import Utilities #from Utilities import numericalSort

#SKLearn
import skimage.io as sk
from sklearn.model_selection import train_test_split, KFold


label_map = {'chris':0, 'claire':1}
# move all data and model to GPU if available
my_device = "cuda" if torch.cuda.is_available() else "cpu"

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet18(img_channel=1, num_classes=2):
    return ResNet(block, [2, 2, 2, 2], img_channel, num_classes)

def ResNet50(img_channel=1, num_classes=2):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=2):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=2):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


# as per datasets requirement
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, labelsFile, rootDir, sourceTransform = None, targetTransform = None, FFGEI = False):
        self.data = pd.read_csv(labelsFile)
        self.rootDir = rootDir
        self.FFGEI = FFGEI
        self.sourceTransform = sourceTransform
        self.targetTransform = targetTransform
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagePath = self.rootDir + "/" + str(self.data['ID'][idx]) + ".jpg"
        image = sk.imread(imagePath)
        image = Image.fromarray(image)
        image = np.asarray(image)
        #IF FFGEI, pre-flatten image
        if self.FFGEI:
            #Transform into tiles
            tiles = Utilities.get_tiles(image)
            #Append them all together
            flat_img = tiles[0]
            for i, t in enumerate(tiles):
                if i > 0:
                    flat_img = np.concatenate([flat_img,tiles[i]])
            image = flat_img

        label = self.data['Class'][idx]

        if self.sourceTransform:
            image = self.sourceTransform(image)

        return image, label

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def create_dataloaders(sourceTransform, targetTransform, labels, images, sizes, batch_size, FFGEI = False):
    #Initialise data paths and check sizes
    os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))
    dataset = CustomDataset(labels, images, sourceTransform, targetTransform, FFGEI)
    df = pd.read_csv(sizes, sep=',',header=None)
    instance_sizes = df.values

    #Split 80% training from the data, this 80% will make up the training and validation datasets
    num_instances = len(instance_sizes)
    train_instances = int(num_instances * 0.7)
    test_instances = num_instances - train_instances
    rng = default_rng()
    #Cut both classes evenly so both the train and test sets have a roughly equal distribution of examples
    class_0_indices = random.sample(range(0, int(num_instances/2)-1), int(test_instances/2))
    class_1_indices = random.sample(range(int(num_instances/2), num_instances), int(test_instances/2))
    test_indices = np.concatenate([class_0_indices, class_1_indices], axis=0)

    #Transform these indices from indices 0-39 (number of instances) to 0-4099 (number of total frames among all instances)
    true_train_indices = []
    true_test_indices = []
    #Test indices
    start_value = 0
    for iter, (index, length) in enumerate(instance_sizes):
        if iter in test_indices:
            for j in range(int(start_value), int(start_value) + int(length)):
                true_test_indices.append(j)
        start_value += int(length)

    #Remainder is training indices
    for i in range(0, sum(map(sum, instance_sizes))):
        if i not in true_test_indices:
            true_train_indices.append(i)

    #Pass the indices through as usual to create the subsets
    print(true_train_indices)
    print(true_test_indices)
    train_data = torch.utils.data.Subset(dataset, true_train_indices)
    test_data = torch.utils.data.Subset(dataset, true_test_indices)
    print(type(true_train_indices), len(true_train_indices), "length here??", len(dataset))
    #for idx, (data, image) in enumerate(train_data):
    #    print("trying once")
    #    print(idx)

    ##Create dataloaders for training/validation set and test set.
    train_val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_data, test_loader

def train_network(data_loader, test_loader, epoch, batch_size, out_path, model_path):

    #Results list (empty 2D array apart from titles
    results = [['Epoch', 'Train_Acc', 'Train_Conf', 'Train_Prec', 'Train_Recall', 'Train_f1', 'T_TP', 'T_FP','T_TN', 'T_FN',
                'Val_Acc', 'Val_Conf', 'Val_Prec', 'Val_Recall', 'Val_f1', 'V_TP', 'V_FP','V_TN', 'V_FN',
                'Test_Acc', 'Test_Conf', 'Test_Prec', 'Test_Recall', 'Test_f1', 'TE_TP', 'TE_FP','TE_TN', 'TE_FN']]
    # Hyperparameters
    in_channels = 1
    num_classes = 2
    learning_rate = 0.001
    num_epochs = epoch

    # Initialize network
    model = ResNet18(img_channel=1, num_classes=num_classes)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    k_folds = 3
    # Set fixed random number seed
    torch.manual_seed(42)
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')
    # K-fold Cross Validation model evaluation, splits train/validation data
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data_loader)):
        fold_results = []
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            data_loader,
            batch_size=batch_size, sampler=train_subsampler)
        valloader = torch.utils.data.DataLoader(
            data_loader,
            batch_size=batch_size, sampler=val_subsampler)

        print("----------------------------------------------------------------type: ", type(trainloader))
        # Init the neural network
        network = ResNet18(img_channel=1, num_classes=num_classes)
        network.apply(reset_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):
            result_row = [epoch + 1]
            print(f'Starting epoch {epoch + 1}')
            # Set current loss value
            current_loss = 0.0
            # Iterate over the DataLoader for training data
            for i, (data, targets) in enumerate(tqdm(trainloader)):
                # Get inputs
                #inputs, targets = data
                data = data.to(device=my_device)
                targets = targets.to(device=my_device)
                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                # backward
                optimizer.zero_grad()
                loss.backward()
                # gradient descent or adam step
                optimizer.step()

            print("epoch: ", epoch)
            #This is the data this fold of the model was just trained on
            result_row = np.concatenate((result_row, copy.deepcopy(evaluate_model(trainloader, model))), axis=0)
            #This is the validation data
            result_row = np.concatenate((result_row, copy.deepcopy(evaluate_model(valloader, model))), axis=0)
            #This is the unseen test data
            result_row = np.concatenate((result_row, copy.deepcopy(evaluate_model(test_loader, model))), axis=0)
            fold_results.append(result_row)

        print("training completed, adding means and standard deviations")
        frame = pd.DataFrame(fold_results)
        means = []
        stds = []
        #Remove column labels
        tmp = frame.iloc[1: , :]
        for column in tmp:
            stds.append(np.std(tmp[column]))
            means.append(tmp[column].mean())

        means[0] = 'Means'
        stds[0] = 'St.Devs'
        fold_results.append(means)
        fold_results.append(stds)
        results = np.concatenate((results, copy.deepcopy(fold_results)), axis=0)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model
        os.makedirs(model_path, exist_ok=True)
        save_path = model_path + '/model_fold_' + str(fold) + '.pth'
        torch.save(network.state_dict(), save_path)

    frame = pd.DataFrame(results)
    #Save as CSV all results
    os.makedirs(out_path, exist_ok=True)
    frame.to_csv(out_path + "results.csv")
    return model

# Check accuracy on training & test to see how good our model
def evaluate_model(loader, model, debug = False):
    num_correct = 0
    num_samples = 0
    chris_confidence = 0 # class 0
    claire_confidence = 0 # class 1
    num_chris = 0
    num_claire = 0
    num_correct_claire = 0
    num_correct_chris = 0

    model.eval()

    # Calculate precision and recall
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0
    prediction_array = []

    with torch.no_grad():
        for x, y in loader:
            #print("what im working with: ", x, y )
            x = x.to(device=my_device)
            y = y.to(device=my_device)
            scores = model(x)

            #Get prediction probabilities
            probs = torch.nn.functional.softmax(scores, dim=1)
            top_p, top_class = probs.topk(1, dim=1)
            _, predictions = scores.max(1)
            #Iterate through results to get TP, FP, FN and FP for various metric calculations
            zipped = zip(y, predictions, probs)
            for i, j, k in zipped:
                if i == j:
                    num_correct+=1
                    if i ==0:
                        num_correct_chris+=1
                        chris_confidence += k[0].item()
                        true_neg += 1
                    else:
                        num_correct_claire+=1
                        claire_confidence += k[1].item()
                        true_pos += 1
                if i.item() == 0:
                    num_chris+=1
                else:
                    num_claire+=1
                if i != j:
                    if j == 0:
                        false_neg += 1
                    elif j == 1:
                        false_pos += 1

            #Also save the predictions in order for the video test
            for pred in predictions:
                prediction_array.append(pred.item())
            #print("predictions: ", predictions, predictions.size(0), )
            #print(predictions.item())
            #prediction_array.append(predictions.item())
            num_samples += predictions.size(0)


    model.train()
    total_chris_confidence = 0
    total_claire_confidence = 0

    #Calculate confidence of each person, given that they appear at all in the testing set
    if num_claire > 0:
        total_claire_confidence = claire_confidence/num_claire * 100
    if num_chris > 0:
        total_chris_confidence = chris_confidence/num_chris * 100

    #Prevent division by 0 errors when calculating precision and recall
    if true_pos > 0 or false_pos > 0:
        precision = true_pos / (true_pos + false_pos)
    else:
        precision = 0

    if true_pos > 0 or false_neg > 0:
        recall = true_pos / (true_pos + false_neg)
    else:
        recall = 0

    #F1 score
    if precision > 0 or recall > 0:
        f1_score = 2 * ((precision * recall)/(precision + recall))
    else:
        f1_score = 0

    total_confidence = 0
    #Prevent division by 0 confidence score errors when using this function for the live video test, as there will only be 1 class present in the data.
    if num_claire > 0 and num_chris > 0:
        total_confidence = ((claire_confidence / num_claire) + (chris_confidence / num_chris)) * 100 / 2
        #print("total prediction confidence: {:.2f}%".format(((claire_confidence/num_claire) + (chris_confidence/num_chris)) * 100 / 2))
    elif num_claire > 0:
        #print("this confidence", claire_confidence, num_claire)
        total_confidence = (claire_confidence / num_claire) * 100 / 2
    elif num_chris > 0:
        #print("that confidence", chris_confidence, num_chris)
        total_confidence = (chris_confidence / num_chris) * 100 / 2

    print("accuracy: {:.2f}".format(num_correct/num_samples * 100))
    print("TP, FP, TN, FN: ", true_pos, false_pos, true_neg, false_neg)
    total_accuracy = num_correct/num_samples * 100

    #If debug, return the prediction array as this is the live video test.
    if debug == False:
        return [total_accuracy, total_confidence, precision, recall, f1_score, true_pos, false_pos, true_neg, false_neg]
    else:
        return [prediction_array, total_accuracy, total_confidence, precision, recall, f1_score]


