# Imports
import torch
import torchvision # torch package for vision related things
import torch.nn as nn
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
import pandas as pd
import skimage.io as sk
#from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageTk, Image# Need to override __init__, __len__, __getitem__
import os
import numpy as np
import cv2
from torchvision.transforms import ToTensor, Lambda
import pandas as pd

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


def ResNet50(img_channel=1, num_classes=2):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=2):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=2):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


# as per datasets requirement
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, labelsFile, rootDir, sourceTransform = None, targetTransform = None):
        self.data = pd.read_csv(labelsFile)
        self.rootDir = rootDir
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

        label = self.data['class'][idx]

        if self.sourceTransform:
            image = self.sourceTransform(image)

        return image, label


def dataloader_test(sourceTransform, targetTransform):
    os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))
    dataset = CustomDataset('./Labels/labels.csv', './Images/GEI/SpecialSilhouettes/test', sourceTransform, targetTransform)
    train_data, test_data = random_split(dataset, [32, 10], generator=torch.Generator().manual_seed(12))
    batch_size = 3
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)#, num_workers=0)
    return train_loader, test_loader

#        tmp = debug_transform(image[0])
#        tmp = np.asarray(tmp)
#        print("retreived label: ", str(image[1]))
#        print("image itself: ", type(image[0]), image[0].shape, type(tmp))
#        cv2.imshow("loaded data: " + str(image[1]), tmp)
#        cv2.waitKey(0)
#        print("sucess")

def train_network(train_data, test_data, out_path):

    #Results list (empty 2D array apart from titles
    results = [['Epoch', 'Train_Acc', 'Train_Conf', 'Train_Prec', 'Train_Recall', 'Train_f1', 'Test_Acc', 'Test_Conf', 'Test_Prec', 'Test_Recall', 'Test_f1']]
    # Hyperparameters
    in_channels = 1
    num_classes = 2
    learning_rate = 0.001
    batch_size = 3
    num_epochs = 10
    
    # Load Data
    #This is just training for now, dataloader is the relevant variable
    #Passes in dataloaders already done
    train_loader = train_data
    test_loader = test_data

    # Initialize network
    model = ResNet50(img_channel=1, num_classes=num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train Network
    for epoch in range(num_epochs):
        result_row = [epoch]
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible

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
        print("training")
        result_row += check_accuracy(train_loader, model)
        print("testing", result_row)
        result_row += check_accuracy(test_loader, model)
        print("results: ", result_row)
        results.append(result_row)
        print("results in total: ", results)
    frame = pd.DataFrame(results)
    print(frame.head())

    #Save as CSV all results
    os.makedirs(out_path, exist_ok=True)
    frame.to_csv(out_path + 'results.csv')
    return model

    # Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
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

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=my_device)
            y = y.to(device=my_device)

            scores = model(x)

            #Get prediction probabilities
            probs = torch.nn.functional.softmax(scores, dim=1)
            top_p, top_class = probs.topk(1, dim=1)
            _, predictions = scores.max(1)

            zipped = zip(y, predictions, probs)
            for i, j, k in zipped:
                if i == j:
                    num_correct+=1
                    if i ==0:
                        num_correct_chris+=1
                        chris_confidence += k[0].item()
                    else:
                        num_correct_claire+=1
                        claire_confidence += k[1].item()
                        true_pos += 1
                if i == 0:
                    num_chris+=1
                elif i == 1:
                    num_claire+=1

                if i != j:
                    if j == 0:
                        false_neg += 1
                    elif j == 1:
                        false_pos += 1


            num_samples += predictions.size(0)


    model.train()

    total_claire_confidence = claire_confidence/num_claire * 100
    total_chris_confidence = chris_confidence/num_chris * 100

    if true_pos > 0 or false_pos > 0:
        print("calcing precision")
        precision = true_pos / (true_pos + false_pos)
    else:
        precision = 0

    if true_pos > 0 or false_neg > 0:
        recall = true_pos / (true_pos + false_neg)
    else:
        recall = 0

    if precision > 0 or recall > 0:
        f1_score = 2 * ((precision * recall)/(precision + recall))
    else:
        f1_score = 0
    print("true pos: ", true_pos, false_neg, false_pos)
    print("chris examples: ", num_chris, " claire examples: ", num_claire)
    print("correct predictions: Chris: {}, Claire: {} ".format(num_correct_chris, num_correct_claire))
    print("precision: {}, recall: {}".format(precision, recall))
    print("total prediction confidence: {:.2f}%".format(((claire_confidence/num_claire) + (chris_confidence/num_chris)) * 100 / 2))
    print("accuracy: {:.2f}".format(num_correct/num_samples * 100))

    total_accuracy = num_correct/num_samples * 100
    total_confidence = ((claire_confidence/num_claire) + (chris_confidence/num_chris)) * 100 / 2
    return [total_accuracy, total_confidence, precision, recall, f1_score]


