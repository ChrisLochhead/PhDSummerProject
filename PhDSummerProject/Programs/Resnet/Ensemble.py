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

#SKLearn
import skimage.io as sk
from sklearn.model_selection import train_test_split, KFold


#Function to split data into n chunks
#add labels path and create them too for all the new data, create sizes 
#Number of folds must be divisible by number of instances: 6 instances means i can have 6, 3 or 2 folds.
def split_data_n_folds(num_folds, sourceTransform, targetTransform, sizes, batch_size, FFGEI, data_path, label_path):
#def create_dataloaders(sourceTransform, targetTransform, labels, images, sizes, batch_size, FFGEI=False):
    # Initialise data paths and check sizes
    k_folds = 2

    #os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))
    dataset = LocalResnet.CustomDataset(label_path, data_path, sourceTransform, targetTransform, FFGEI)
    #dataset = CustomDataset(labels, images, sourceTransform, targetTransform, FFGEI)
    print("length here?", len(dataset))
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


    print("length 1", len(dataset))
    for n in range(k_folds):

        # For n in num folds, make an array of indices that are non repeating to randomly split the instances
        fold_indices = []
        # Indices from 0 - num_instances -1
        possible_indices = list(range(0, num_instances))
        print("num instances: ", possible_indices)
        print("fold size and possible indices", len(possible_indices), fold_size)
        #array of random numbers of size fold_size
        found_unique = False
        while found_unique == False:
            #print("found unique? ", found_unique)
            selected_indices = rng.choice(len(possible_indices), size=fold_size, replace=False)
            new_indices = []
            for i in selected_indices:
                #print("indice: ", i, len(possible_indices))
                new_indices.append(possible_indices[i])

            #print("num new indices: ", len(new_indices), new_indices, len(possible_indices), possible_indices)
            if set(new_indices).issubset(set(possible_indices)):
                #print("result: ", new_indices, possible_indices)
                possible_indices = [x for x in possible_indices if x not in new_indices]
                #print("possible indices after removal", possible_indices)
                fold_indices.append(new_indices)
                if len(possible_indices) == 0:
                    found_unique = True
                else:
                    #print("current length: ", len(possible_indices))
                    #if remainder is less than size of a new fold, just add it to the latest
                    if len(possible_indices) < fold_size:
                        for i in possible_indices:
                            fold_indices[-1].append(i)
                        found_unique = True
                        #print("found unique??", found_unique)
        print("loop completed")
        #Should be 3 here only 1 atm
        #for n in fold_indices:
        #    print("fold: ", n)

    print("length 2", len(dataset))
    #Assuming we now have n folds of indices, transform each into actual values
    true_fold_indices = []
    if num_folds > 1:
        for n in range(num_folds):
            empty = []
            true_fold_indices.append(empty)
    #print("true fold type: ", true_fold_indices[0][0], len(true_fold_indices[0][0]))
    # Transform these indices from indices 1-42 (number of instances) to 0-4099 (number of total frames among all instances)
    start_value = 0
    for iter, (index, length) in enumerate(instance_sizes):
        #print("getting true instances")
        for fold_index, fold in enumerate(fold_indices):
            #print("in new fold", fold)
            if iter in fold:
                true_fold = []
                #new_fold = []
                for j in range(int(start_value), int(start_value) + int(length)):
                    true_fold.append(j)
                    #print("appending: ", j, index, length)
                    #new_fold.append(j)
                start_value += int(length)
                if num_folds > 1:
                    true_fold_indices[fold_index].append(true_fold)
                else:
                    true_fold_indices.append(true_fold)

    print("assuming this works: ")
    total = 0
    if num_folds >1:
        for fold in true_fold_indices:
            #print("true fold: ", fold)
            total+= len(fold)
            print("length = ", len(fold))
        print("total instances recorded: ", total)
        print("this should be equal to the total files in the 6 few shot instances.")


    print("length 3", len(dataset))
    print("unravelling", true_fold_indices)
    unravelled_fold_indices = []
    if num_folds > 1:
        for i, fold in enumerate(true_fold_indices):
            if len(fold) > 1:
                unravelled_fold = []
                for f in fold:
                    unravelled_fold = unravelled_fold + f
                    #print("unravelled_fold: ", unravelled_fold)
                unravelled_fold_indices.append(unravelled_fold)
            else:
                unravelled_fold_indices = fold
        for fold in unravelled_fold_indices:
            print("finished fold", fold)
    else:
        for f in true_fold_indices:
            unravelled_fold_indices += f
        print("total indices: ", unravelled_fold_indices)


    # Debug
    # print("lengths - all, train, test: ", sum(instance_sizes), len(true_train_indices), len(true_test_indices))#, len(true_valid_indices))
    # print("test indices", len(true_test_indices))
    # print("train indices, ", len(true_train_indices))

    #Subset each fold into train and test data
    print("length 4", len(dataset))
    folded_train_data = []
    folded_test_data = []
    if num_folds > 1:
        for iter, fold in enumerate(unravelled_fold_indices):
            print("------------------- fold : ", iter)
            #Split true indices in half, give half to train and half to test
            num_examples = len(unravelled_fold_indices[iter])
            cut_point = int(num_examples * 0.7)
            print("e: ", unravelled_fold_indices[iter][:cut_point])
            print("g: ", unravelled_fold_indices[iter][cut_point:])
            #example = [1,2,3,4,5,6,7,8,9]
            print("length before changing", len(dataset))
            #fold_train = torch.utils.data.Subset(dataset, example)
            fold_train = torch.utils.data.Subset(dataset, unravelled_fold_indices[iter][:cut_point])
            fold_test = torch.utils.data.Subset(dataset, unravelled_fold_indices[iter][cut_point:])
            print("fail here", len(fold_train))
            #for idx, (data, image) in enumerate(fold_train):
            #    print("trying once")
            #    print(idx)
            folded_train_data.append(fold_train)
            folded_test_data.append(fold_test)
    
        print("if this is working, each fold of train and test should add up to the total")
        for i in folded_train_data:
            print("folded train:", len(i), i)
        for i in folded_test_data:
            print("folded test:", len(i), i)
    else:
        num_examples = len(unravelled_fold_indices)
        cut_point = int(num_examples * 0.5)
        print("e: ", unravelled_fold_indices[:cut_point])
        print("g: ", unravelled_fold_indices[cut_point:])
        # example = [1,2,3,4,5,6,7,8,9]
        print("length before changing", len(dataset))
        # fold_train = torch.utils.data.Subset(dataset, example)
        fold_train = torch.utils.data.Subset(dataset, unravelled_fold_indices[:cut_point])
        fold_test = torch.utils.data.Subset(dataset, unravelled_fold_indices[cut_point:])
        print("fail here", fold_train, fold_test)
        # for idx, (data, image) in enumerate(fold_train):
        #    print("trying once")
        #    print(idx)
        #folded_train_data.append(fold_train)
        #folded_test_data.append(fold_test)


    # Pass the indices through as usual to create the subsets
    #train_data = torch.utils.data.Subset(dataset, true_train_indices)
    #test_data = torch.utils.data.Subset(dataset, true_test_indices)

    # Create dataloaders for training/validation set and test set.
    train_loader_array = []
    test_loader_array = []
    if num_folds > 1:
        for i in range(num_folds):
            print(len(folded_train_data[i]))
            print("f: ", len(folded_test_data[i]))
            train_loader_array.append(torch.utils.data.DataLoader(folded_train_data[i], batch_size=batch_size, shuffle=True))
            test_loader_array.append(torch.utils.data.DataLoader(folded_test_data[i], batch_size=batch_size, shuffle=True))
    else:
        print("Doing this?????", fold_train, len(fold_train))
        train_loader_array.append(torch.utils.data.DataLoader(fold_train, batch_size=1, shuffle=True))
        test_loader_array.append(torch.utils.data.DataLoader(fold_test, batch_size=1, shuffle=True))
    
    print("christ it worked?", len(train_loader_array[0]), len(test_loader_array[0]))

    for idx, (data, image) in enumerate(train_loader_array[0]):
        print("trying twice")
        print(idx, image)
    #Array of arrays same number of dimensions as number of folds
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
def create_ensemble_resnets(num_models, model_path, empty = False):
    #Create n models from preloaded path
    models = []
    for n in range(num_models):
        if empty == False:
            model = load_model(model_path)
        else:
            model = ResNet50(img_channel=1, num_classes=2)
            
        models.append(model)

    return models

def train_ensemble_model(training_data, testing_data, models, epoch, batch_size, results_out, model_out):
    # Results list (empty 2D array apart from titles
    results = [['Epoch', 'Train_Acc', 'Train_Conf', 'Train_Prec', 'Train_Recall', 'Train_f1',
                'Val_Acc', 'Val_Conf', 'Val_Prec', 'Val_Recall', 'Val_f1',
                'Test_Acc', 'Test_Conf', 'Test_Prec', 'Test_Recall', 'Test_f1']]
    # Hyperparameters
    in_channels = 1
    num_classes = 2
    learning_rate = 0.001
    num_epochs = epoch
    k_folds = 2


    ##Training begins
    print("is it even in here??")
    print("training data: ", len(training_data))
    #kfold = KFold(n_splits=k_folds, shuffle=True)
    for iterator, model in enumerate(models):

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Start print
        print('--------------------------------', len(training_data[iterator]), type(training_data[iterator]))
        print("model exist? ", model)
        #for idx, (data, image) in enumerate(training_data[0]):
        #    print("trying twice")
        #    print(idx, image)
        # K-fold Cross Validation model evaluation, splits train/validation data
        #for fold, (train_ids, val_ids) in enumerate(kfold.split(training_data[0])):
        for fold, value in enumerate(training_data[0]):

            # Creating data indices for training and validation splits:
            dataset_size = len(training_data[0])
            indices = list(range(dataset_size))
            testing_size = len(testing_data[0])
            test_indices = list(range(testing_size))

            print("dataset size: ", dataset_size, " indices: ", indices)
            split = int(np.ceil(0.3 * dataset_size))
            print("split: ", split)
            np.random.seed(13)
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            print("train and val indices: ", train_indices, val_indices)

            print("is it actually making it here?")

            fold_results = []
            #print("Fold number: ", fold)
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)#train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_indices)#(val_ids)

            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(
                training_data[0],
                batch_size=batch_size, sampler=train_subsampler)
            valloader = torch.utils.data.DataLoader(
                training_data[0],
                batch_size=batch_size, sampler=val_subsampler)
            test_loader = torch.utils.data.DataLoader(
                testing_data[0],
                batch_size=batch_size, sampler=test_indices)

            print("----------------------------------------------------------------type: ", type(train_loader), type(training_data[0]))
            print("type: ", type(tqdm(train_loader.dataset)))
            # Init the neural network
            # network = ResNet50(img_channel=1, num_classes=num_classes)
            # network.apply(reset_weights)
            # criterion = nn.CrossEntropyLoss()
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Run the training loop for defined number of epochs
            for epoch in range(num_epochs):
                result_row = [epoch + 1]
                print(f'Starting epoch {epoch + 1}')
                # Set current loss value
                current_loss = 0.0
                # Iterate over the DataLoader for training data
                for i, (data, targets) in enumerate(tqdm(train_loader.dataset)):
                    # Get inputs
                    # inputs, targets = data
                    data = data.to(device=LocalResnet.my_device)
                    targets = targets.to(device=LocalResnet.my_device)
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
                result_row = np.concatenate((result_row, copy.deepcopy(evaluate_step(train_loader, model))), axis=0)
                # This is the validation data
                result_row = np.concatenate((result_row, copy.deepcopy(evaluate_step(valloader, model))), axis=0)
                # This is the unseen test data
                result_row = np.concatenate((result_row, copy.deepcopy(evaluate_step(test_loader, model))), axis=0)
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
            save_ensembles(model, model_out, iterator, str(fold))

    frame = pd.DataFrame(results)

    # Save as CSV all results
    os.makedirs(results_out, exist_ok=True)
    frame.to_csv(results_out + "results.csv")
    return models


# Check accuracy on training & test to see how good our model
def evaluate_step(loader, model, debug = False):
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
    prediction_array = []

    with torch.no_grad():
        for x, y in loader.dataset:
            print("x and y: ", x, y)
            #print("what im working with", len(x), len(y), type(x), type(y))
            x = x.to(device=LocalResnet.my_device)
            y = y.to(device=LocalResnet.my_device)
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

    #Debug
    #print("chris examples: ", num_chris, " claire examples: ", num_claire)
    #print("correct predictions: Chris: {}, Claire: {} ".format(num_correct_chris, num_correct_claire))
    #print("precision: {}, recall: {}".format(precision, recall))

    total_confidence = 0
    #Prevent division by 0 confidence score errors when using this function for the live video test, as there will only be 1 class present in the data.
    if num_claire > 0 and num_chris > 0:
        total_confidence = ((claire_confidence / num_claire) + (chris_confidence / num_chris)) * 100 / 2
        print("total prediction confidence: {:.2f}%".format(((claire_confidence/num_claire) + (chris_confidence/num_chris)) * 100 / 2))
    elif num_claire > 0:
        print("this confidence", claire_confidence, num_claire)
        total_confidence = (claire_confidence / num_claire) * 100 / 2
    elif num_chris > 0:
        print("that confidence", chris_confidence, num_chris)
        total_confidence = (chris_confidence / num_chris) * 100 / 2

    print("accuracy: {:.2f}".format(num_correct/num_samples * 100))
    total_accuracy = num_correct/num_samples * 100

    #If debug, return the prediction array as this is the live video test.
    if debug == False:
        return [total_accuracy, total_confidence, precision, recall, f1_score]
    else:
        return [prediction_array, total_accuracy, total_confidence, precision, recall, f1_score]


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
            for x, y in loader.dataset:
                x = x.to(device=LocalResnet.my_device)
                y = y.to(device=LocalResnet.my_device)
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
        # If debug, return the prediction array as this is the live video test.
        if debug == False:
            return [total_accuracy, total_confidence, precision, recall, f1_score]
        else:
            return [prediction_array, total_accuracy, total_confidence, precision, recall, f1_score]




#Function to save resnets
def save_ensembles(model, model_path, ensemble_count, fold_count):
    # Saving the model
    os.makedirs(model_path, exist_ok=True)
    save_path = model_path + 'ensemble_model_' + str(ensemble_count) + '/model_fold_' + str(fold_count) + '.pth'
    torch.save(model.state_dict(), save_path)

#Function to load resnets
def load_model(model_path):
    model = LocalResnet.ResNet50(img_channel=1, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

#Main function
def few_shot_ensemble_experiment(n, batch_size, epoch, standard_data = './Images/GEI/SpecialSilhouettes', few_shot_data = './Images/GEI/FewShot/SpecialSilhouettes'):
    batch_size = 3
    epoch = 15
    target = Lambda( lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    #Split regular data by n-folds
    training_data_array, testing_data_array = split_data_n_folds(num_folds=n,
                                                            sourceTransform=ToTensor(),
                                                            targetTransform=target,
                                                            sizes='./Instance_Counts/normal/GEI.csv',
                                                            batch_size=batch_size,
                                                            FFGEI=False,
                                                            data_path='./Images/GEI/Masks',
                                                            label_path='./labels/labels.csv')
    
    #Create array of Resnets and train them on the standard data, this will save their progress and iterim results too
    #ensemble_models = create_ensemble_resnets(n, training_data_array, testing_data_array)

    ensemble_models =  create_ensemble_resnets(n, model_path='./Models/GEI_SpecialSilhouettes/model_fold_2.pth')

    #Split the few shot data
    #Split regular data by n-folds
    print("I AM ACTUALLY MAKING IT HERE ------------------------------------------------------------------------")
    few_shot_training, few_shot_testing = split_data_n_folds(num_folds=1,
                                                            sourceTransform=ToTensor(),
                                                            targetTransform=target,
                                                            sizes='./Instance_Counts/FewShot/Normal/GEI.csv',
                                                            batch_size=batch_size,
                                                            FFGEI=False,
                                                            data_path='./Images/GEI/FewShot/SpecialSilhouettes',
                                                            label_path='./labels/FewShot/labels.csv')

    #Currently training is done in the creation, split into the train, single model
    ensemble_models = train_ensemble_model(training_data = few_shot_training, testing_data = few_shot_testing, models = ensemble_models,
                                           epoch = 15, batch_size = 1,
                                           results_out ='./Results/Ensemble/GEI_SpecialSilhouettes/',
                                           model_out = './Models/Ensemble/GEI_SpecialSilhouettes/model.pth' )
        
    #Evaluate ensemble resnets and conduct voting for classification
    evaluation_results = evalutate_ensemble(few_shot_testing, ensemble_models)

    empty_models = def create_ensemble_resnets(n, None, empty = True)
    #Evaluate standard models with no pre-trained weights as a control group
    control_results = evalutate_ensemble(few_shot_testing, empty_models)


    #Save the control and evaluation results (together maybe?) after that clean the code and we are all done :) 
    #print("training completed, adding means and standard deviations")
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

