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
from PIL import ImageTk, Image
import os
import numpy as np
import cv2
import LocalResnet
#Local files
import Utilities #from Utilities import numericalSort


#Function to split data into n chunks
#add labels path and create them too for all the new data, create sizes 
#Number of folds must be divisible by number of instances: 6 instances means i can have 6, 3 or 2 folds.
def split_data_n_folds(num_folds, sourceTransform, targetTransform, sizes, batch_size, data = None, FFGEI = False, datapath = 'None', labelpath = 'None'):
#def create_dataloaders(sourceTransform, targetTransform, labels, images, sizes, batch_size, FFGEI=False):
    # Initialise data paths and check sizes
    num_folds = 5

    #os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))
    dataset = CustomDataset(labels, images, sourceTransform, targetTransform, FFGEI)
    df = pd.read_csv(sizes, sep=',', header=None)
    instance_sizes = df.values

    # Split 80% training from the data, this 80% will make up the training and validation datasets
    num_instances = len(instance_sizes)
    train_instances = int(num_instances * 0.8)
    test_instances = num_instances - train_instances
    rng = default_rng()
    test_indices = rng.choice(num_instances - 1, size=test_instances, replace=False)

    fold_size = int(num_instances/num_folds)
    #assume this fold size is 1
    #For n in num folds, make an array of indices that are non repeating to randomly split the instances
    fold_indices = []
    #Indices from 0 - num_instances -1
    possible_indices = list(range(0, num_instances))
    print("num instances: ", possible_instances)
    for n in num_folds:
        #array of random numbers of size fold_size
        found_unique = False
        while found_unique == False:
            new_indices = rng.choice(num_instances - 1, size=fold_size, replace=False)
            if set(new_indices).issubset(set(possible_indices)):
                print("result: ", new_indices, possible_indices)
                possible_indices = [x for x in new_indices if x not in possible_indices]
                print("possible indices after removal", possible_indices)
                fold_indices.append(new_indices)
                found_unique = True
        print("loop completed")
        for n in fold_indices:
            print("fold: ", n)

    #Assuming we now have n folds of indices, transform each into actual values
    true_fold_indices = []
    # Transform these indices from indices 1-42 (number of instances) to 0-4099 (number of total frames among all instances)
    start_value = 0
    for iter, (index, length) in enumerate(instance_sizes):
        for fold in num_folds:
            if iter in fold:
                new_fold = []
                for j in range(int(start_value), int(start_value) + int(length)):
                    new_fold.append(j)
                start_value += int(length)
                true_fold_indices.append(new_fold)

    print("assuming this works: ")
    total = 0
    for fold in true_fold_indices:
        total+= len(fold)
        print("length = ", len(fold))
    print("total instances recorded: ", total)
    print("this should be equal to the total files in the 6 few shot instances.")


    # Debug
    # print("lengths - all, train, test: ", sum(instance_sizes), len(true_train_indices), len(true_test_indices))#, len(true_valid_indices))
    # print("test indices", len(true_test_indices))
    # print("train indices, ", len(true_train_indices))

    #Subset each fold into train and test data
    folded_train_data = []
    folded_test_data = []
    for iter, fold in enumerate(num_folds):
        #Split true indices in half, give half to train and half to test
        num_examples = len(true_fold_indices[iter])
        half_point = int(num_examples/2)

        fold_train = torch.utils.data.Subset(dataset, true_fold_indices[iter][:half_point])
        fold_test = torch.utils.data.Subset(dataset, true_fold_indices[iter][half_point:])
        folded_train_data.append(fold_train)
        folded_test_data.append(fold_test)
    
    print("if this is working, each fold of train and test should add up to the total")
    for i in folded_train_data:
        print("folded train:", len(i))
    for i in folded_test_data:
        print("folded test:", len(i))
    # Pass the indices through as usual to create the subsets
    #train_data = torch.utils.data.Subset(dataset, true_train_indices)
    #test_data = torch.utils.data.Subset(dataset, true_test_indices)

    # Create dataloaders for training/validation set and test set.
    train_loader_array = []
    test_loader_array = []
    for i, fold in enumerate(num_folds):
        train_loader_array = torch.utils.data.DataLoader(folded_train_data[i], batch_size=batch_size, shuffle=True)
        test_loader_array = torch.utils.data.DataLoader(folded_test_data[i], batch_size=batch_size, shuffle=True)
    
    print("christ it worked?")
    return train_loader_array, test_loader_array
    #train_val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    #return train_data, test_loader
    print("datasets prepared sucessfully")
    #model = LocalResnet.train_network(train_val_loader, test_loader, epoch=epoch, batch_size=batch_size,
    #                                  out_path='./Results/FFGEI/GraphCut/', model_path='./Models/FFGEI_GraphCut/')


    #return 2D array of data split into train and test of length n

#Function to create n resnets and train them on n chunks
#For each experiment, append model and results tabs with type
def create_ensemble_resnets(num_models, training_data, testing_data, epoch, batch_size, out_path = './Results/FewShot/FFGEI_Mask/', modelPath = './Models/FewShot/FFGEI_Mask/model_fold_2.pth'):
    #Create 5 models
#def train_network(data_loader, test_loader, epoch, batch_size, out_path, model_path):

    # Results list (empty 2D array apart from titles
    results = [['Epoch', 'Train_Acc', 'Train_Conf', 'Train_Prec', 'Train_Recall', 'Train_f1',
                'Val_Acc', 'Val_Conf', 'Val_Prec', 'Val_Recall', 'Val_f1',
                'Test_Acc', 'Test_Conf', 'Test_Prec', 'Test_Recall', 'Test_f1']]
    # Hyperparameters
    in_channels = 1
    num_classes = 2
    learning_rate = 0.001
    num_epochs = epoch

    # Initialize network
    models = []
    for n in num_models:
        #load in most developed model on the standard data
        model = load_model(model_path)
        #model = ResNet18(img_channel=1, num_classes=num_classes)
        #model.apply(reset_weights)
        # Loss and optimizer
        #criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #k_folds = 3
        # Set fixed random number seed
        #torch.manual_seed(42)
        # Define the K-fold Cross Validator
        #kfold = KFold(n_splits=k_folds, shuffle=True)
        models.append(model)
        
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for iterator, model in enumerate(models):
        # Start print
        print('--------------------------------')
        # K-fold Cross Validation model evaluation, splits train/validation data
        for fold, (train_ids, val_ids) in enumerate(kfold.split(training_data[iterator])):
            fold_results = []
            print("Fold number: ", fold)
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    
            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(
                training_data[iterator],
                batch_size=batch_size, sampler=train_subsampler)
            valloader = torch.utils.data.DataLoader(
                training_data[iterator],
               batch_size=batch_size, sampler=val_subsampler)
    
            # Init the neural network
            #network = ResNet50(img_channel=1, num_classes=num_classes)
            #network.apply(reset_weights)
            #criterion = nn.CrossEntropyLoss()
            #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
            # Run the training loop for defined number of epochs
            for epoch in range(num_epochs):
                result_row = [epoch + 1]
                print(f'Starting epoch {epoch + 1}')
                # Set current loss value
                current_loss = 0.0
                # Iterate over the DataLoader for training data
                for i, (data, targets) in enumerate(tqdm(train_loader)):
                    # Get inputs
                    # inputs, targets = data
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
                # This is the data this fold of the model was just trained on
                result_row = np.concatenate((result_row, copy.deepcopy(evaluate_model(train_loader, model))), axis=0)
                # This is the validation data
                result_row = np.concatenate((result_row, copy.deepcopy(evaluate_model(valloader, model))), axis=0)
                # This is the unseen test data
                result_row = np.concatenate((result_row, copy.deepcopy(evaluate_model(testing_data, model))), axis=0)
                fold_results.append(result_row)
    
            print("training completed, adding means and standard deviations")
            frame = pd.DataFrame(fold_results)
            means = []
            stds = []
            # Remove column labels
            tmp = frame.iloc[1:, :]
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
            save_ensembles(model, model_path, iterator, str(fold))
            #os.makedirs(model_path, exist_ok=True)
            #save_path = model_path + 'ensemble_model_' + iterator + '/model_fold_' + str(fold) + '.pth'
            #torch.save(model.state_dict(), save_path)

    frame = pd.DataFrame(results)

    # Save as CSV all results
    os.makedirs(out_path, exist_ok=True)
    frame.to_csv(out_path + "results.csv")
    return models



    #For each fold, train the corresponding model on the corresponding data

    #Add metric results after each epoch in the same way as the original
    print("empty")

def train_single_model(training_data, model):
    print("empty")

#Function to evaluate resnets
def evaluate_ensemble(models, testing_data, modelpath = 'None'):
    #Collate the predictions of each model on each testing data item

    #Calculate results via hard-voting

    #Save results as a table in the same way as the training, with the added confidence score in terms of votes
    # Check accuracy on training & test to see how good our model
#def evaluate_model(loader, model, debug=False):
    for model in models:
        num_correct = 0
        num_samples = 0
        chris_confidence = 0  # class 0
        claire_confidence = 0  # class 1
        num_chris = 0
        num_claire = 0
        num_correct_claire = 0
        num_correct_chris = 0

        model.eval()

        # Calculate precision and recall
        true_pos = 0
        false_pos = 0
        false_neg = 0
        prediction_array = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=my_device)
                y = y.to(device=my_device)
                scores = model(x)

                # Get prediction probabilities
                probs = torch.nn.functional.softmax(scores, dim=1)
                top_p, top_class = probs.topk(1, dim=1)
                _, predictions = scores.max(1)
                # Iterate through results to get TP, FP, FN and FP for various metric calculations
                zipped = zip(y, predictions, probs)
                for i, j, k in zipped:
                    if i == j:
                        num_correct += 1
                        if i == 0:
                            num_correct_chris += 1
                            chris_confidence += k[0].item()
                        else:
                            num_correct_claire += 1
                            claire_confidence += k[1].item()
                            true_pos += 1
                    if i.item() == 0:
                        num_chris += 1
                    else:
                        num_claire += 1
                    if i != j:
                        if j == 0:
                            false_neg += 1
                        elif j == 1:
                            false_pos += 1

                # Also save the predictions in order for the video test
                prediction_array.append(predictions.item())
                num_samples += predictions.size(0)

        model.train()
        total_chris_confidence = 0
        total_claire_confidence = 0

        # Calculate confidence of each person, given that they appear at all in the testing set
        if num_claire > 0:
            total_claire_confidence = claire_confidence / num_claire * 100
        if num_chris > 0:
            total_chris_confidence = chris_confidence / num_chris * 100

        # Prevent division by 0 errors when calculating precision and recall
        if true_pos > 0 or false_pos > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0

        if true_pos > 0 or false_neg > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0

        # F1 score
        if precision > 0 or recall > 0:
            f1_score = 2 * ((precision * recall) / (precision + recall))
        else:
            f1_score = 0

        # Debug
        # print("chris examples: ", num_chris, " claire examples: ", num_claire)
        # print("correct predictions: Chris: {}, Claire: {} ".format(num_correct_chris, num_correct_claire))
        # print("precision: {}, recall: {}".format(precision, recall))

        total_confidence = 0
        # Prevent division by 0 confidence score errors when using this function for the live video test, as there will only be 1 class present in the data.
        if num_claire > 0 and num_chris > 0:
            total_confidence = ((claire_confidence / num_claire) + (chris_confidence / num_chris)) * 100 / 2
            print("total prediction confidence: {:.2f}%".format(
                ((claire_confidence / num_claire) + (chris_confidence / num_chris)) * 100 / 2))
        elif num_claire > 0:
            print("this confidence", claire_confidence, num_claire)
            total_confidence = (claire_confidence / num_claire) * 100 / 2
        elif num_chris > 0:
            print("that confidence", chris_confidence, num_chris)
            total_confidence = (chris_confidence / num_chris) * 100 / 2

        print("accuracy: {:.2f}".format(num_correct / num_samples * 100))
        total_accuracy = num_correct / num_samples * 100


        ##Instead of returning, save each table as its own excel file, naming according to which model it is on.
        ##Instead of returning, save each table as its own excel file, naming according to which model it is on.
        # If debug, return the prediction array as this is the live video test.
        if debug == False:
            return [total_accuracy, total_confidence, precision, recall, f1_score]
        else:
            return [prediction_array, total_accuracy, total_confidence, precision, recall, f1_score]




#Function to save resnets
def save_ensembles(model, model_path, ensemble_count, fold_count):
    # Saving the model
    os.makedirs(model_path, exist_ok=True)
    save_path = model_path + 'ensemble_model_' + ensemble_count + '/model_fold_' + fold_count + '.pth'
    torch.save(model.state_dict(), save_path)

#Function to load resnets
def load_model(model_path):
    model = ResNet18(img_channel=1, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

#Main function
def few_shot_ensemble_experiment(n, data_path, few_shot_path):
    #Split training and testing data into 5 folds each
    training_data_array = split_data_n_folds(5, datapath = data_path)
    #few_shot_training, few_shot_testing = split_data_n_folds(5, datapath = few_shot_path)

    batch_size = 3
    epoch = 15
    target = Lambda(
        lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    train_val_loader, test_loader = Ensemble.split_data_n_folds(num_folds=3,
                                                                sourceTransform=ToTensor(),
                                                                targetTransform=target,
                                                                sizes='./Instance_Counts/normal/GEI.csv',
                                                                batch_size=batch_size,
                                                                FFGEI=False,
                                                                datapath='./Images/GEI/SpecialSilhouettes',
                                                                labelpath='./labels/labels.csv')
    
    
    #Create array of Resnets
    ensemble_models = create_ensemble_resnets(5, training_data_array, testing_data_array)
    #Currently training is done in the creation, split into the train, single model
    #for iter, training_set in enumerate(few_shot_training):
    #    ensemble_models[iter] = train_single_model(training_set, ensemble_models[iter])
        
    #Evaluate ensemble resnets and conduct voting for classification
    evaluation_results = evalutate_ensemble(few_shot_testing, ensemble_models)
