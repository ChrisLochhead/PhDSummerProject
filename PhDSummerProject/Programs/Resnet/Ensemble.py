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

    dataset = LocalResnet.CustomDataset(label_path, data_path, sourceTransform, targetTransform, FFGEI)
    print("length here?", len(dataset))
    df = pd.read_csv(sizes, sep=',', header=None)
    instance_sizes = df.values
    rng = np.random.default_rng()
    # Split 80% training from the data, this 80% will make up the training and validation datasets
    num_instances = len(instance_sizes)
    fold_size = int(num_instances/num_folds)

    train_array_folds = []
    test_array_folds = []
    fold_indices = []

    for n in range(num_folds):
        # For n in num folds, make an array of indices that are non repeating to randomly split the instances
        # Indices from 0 - num_instances -1
        fold_set = []
        possible_indices = list(range(0, num_instances))
        #array of random numbers of size fold_size
        found_unique = False
        while found_unique == False:
            selected_indices = rng.choice(len(possible_indices), size=fold_size, replace=False)
            new_indices = []
            for i in selected_indices:
                new_indices.append(possible_indices[i])
        
            if set(new_indices).issubset(set(possible_indices)):
                possible_indices = [x for x in possible_indices if x not in new_indices]
                fold_set.append(new_indices)
                if len(possible_indices) == 0:
                    found_unique = True
                else:
                    #if remainder is less than size of a new fold, just add it to the latest
                    if len(possible_indices) < fold_size:
                        for i in possible_indices:
                            fold_set[-1].append(i)
                        found_unique = True
        fold_indices.append(fold_set)
        print("loop completed")

    print("fold indice stuff: ", fold_indices, len(fold_indices))
    #Assuming we now have n folds of indices, transform each into actual values
    true_fold_indices = []
    #if num_folds > 1:
    for n in range(num_folds):
        empty = np.zeros(6, dtype=object)
        true_fold_indices.append(empty)

    # Transform these indices from indices 1-42 (number of instances) to 0-4099 (number of total frames among all instances)
    for fold_index, fold in enumerate(fold_indices):
        start_value = 0
        print("---new fold")
        #instance sizes - 1- 42
        for iter, (index, length) in enumerate(instance_sizes):
            # folds, each fold f is 0, 1 or 2
            for fold_iter, f in enumerate(fold):
                print("here's f: ", f, fold_iter)
                if iter in f:
                    for i, unit in enumerate(f):
                        if iter == unit:
                            f_position = i
                    true_fold = []
                    for j in range(int(start_value), int(start_value) + int(length)):
                        true_fold.append(j)
                    start_value += int(length)
                    print("appending true fold: ", true_fold, fold_iter, fold_index)
                    true_fold_indices[fold_index][(2 * fold_iter) + f_position] = true_fold

    print("is it working? ", true_fold_indices)
    for a in true_fold_indices:
        for b in a:
            print(b)
        print(a)
    total = 0
    if num_folds >1:
        for fold in true_fold_indices:
            total+= len(fold)
            print("length = ", len(fold))
        print("total instances recorded: ", total)
        print("this should be equal to the total files in the 6 few shot instances.")


    print("unravelling", true_fold_indices)
    unravelled_fold_indices = []

    for i, fold in enumerate(true_fold_indices):
        unravelled_fold = []
        for f in fold:
            unravelled_fold = unravelled_fold + f
        unravelled_fold_indices.append(unravelled_fold)

    print("unravelled: ", unravelled_fold_indices)
    #Subset each fold into train and test data
    folded_train_data = []
    folded_test_data = []
    #if num_folds > 1:
    for iter, fold in enumerate(unravelled_fold_indices):
        #Split true indices in half, give half to train and half to test
        num_examples = len(unravelled_fold_indices[iter])
        cut_point = int(num_examples * 0.5)
        fold_train = torch.utils.data.Subset(dataset, unravelled_fold_indices[iter][:cut_point])
        fold_test = torch.utils.data.Subset(dataset, unravelled_fold_indices[iter][cut_point:])

        folded_train_data.append(fold_train)
        folded_test_data.append(fold_test)

    # Create dataloaders for training/validation set and test set.
    train_loader_array = []
    test_loader_array = []

    for i in range(num_folds):
        print(len(folded_train_data[i]))
        print("f: ", len(folded_test_data[i]))
        print("is it fucked here: ")
        train_loader_array.append(torch.utils.data.DataLoader(folded_train_data[i], batch_size=1, shuffle=True))
        test_loader_array.append(torch.utils.data.DataLoader(folded_test_data[i], batch_size=1, shuffle=True))
        for x, y in train_loader_array[i]:
            print("ys : ", y)


    #Array of arrays same number of dimensions as number of folds
    print("length of this thing?? ", len(train_loader_array))
    #train_array_folds.append(train_loader_array)
    #test_array_folds.append(test_loader_array)

    return train_loader_array, test_loader_array

#Function to create n resnets and train them on n chunks
#For each experiment, append model and results tabs with type
def create_ensemble_resnets(num_models, model_path, empty = False):
    #Create n models from preloaded path
    models = []
    for n in range(num_models):
        if empty == False:
            model = load_model(model_path)
        else:
            model = LocalResnet.ResNet50(img_channel=1, num_classes=2)
            
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


    print("number of models detected: ", len(models))
    ##Training begins
    for iterator, model in enumerate(models):
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Start print
        print('-----------------------------------------------------------------')#model number ', iterator, len(training_data[iterator]), type(training_data[iterator]))
        # K-fold Cross Validation model evaluation, splits train/validation data
        #for fold, (train_ids, val_ids) in enumerate(kfold.split(training_data[0])):
        #for fold, value in enumerate(training_data):#[0]):

        # Creating data indices for training and validation splits:
        dataset_size = len(training_data[iterator])
        indices = list(range(dataset_size))
        testing_size = len(testing_data[iterator])
        #testing_size = len(testing_data[0])
        test_indices = list(range(testing_size))

        split = int(np.ceil(0.3 * dataset_size))

        np.random.seed(13)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]


        fold_results = []
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)#train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_indices)#(val_ids)
        print("whatama working with : ", len(training_data), len(training_data[iterator]))
        print("wtf is this then: ", training_data[0])
        print("this?", training_data)
        for x, y in training_data[0]:
            print("data:", y)

        #print("and this?", training_data[0][0])
        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
            training_data[iterator],
            batch_size=batch_size, sampler=train_subsampler)
        valloader = torch.utils.data.DataLoader(
            training_data[iterator],
            batch_size=batch_size, sampler=val_subsampler)
        test_loader = torch.utils.data.DataLoader(
            testing_data[iterator],
            batch_size=batch_size, sampler=test_indices)

        print("----------------------------------------------------------------type: ", type(train_loader), type(training_data[0]))

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):
            result_row = [epoch + 1]
            print(f'Starting epoch {epoch + 1}')
            # Set current loss value
            current_loss = 0.0
            print("dataset: ", train_loader.dataset)
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
        print("frame:", frame)
        tmp = frame.iloc[1:, :]
        print("temp", tmp)
        for column in tmp:
            #print("calcing stds from : ", tmp[column], np.std(tmp[column]))
            #print("calcing means from: ", tmp[column], tmp[column].mean())
            stds.append(np.std(tmp[column]))
            means.append(tmp[column].mean())

        means[0] = 'Means'
        stds[0] = 'St.Devs'
        fold_results.append(means)
        fold_results.append(stds)
        results = np.concatenate((results, copy.deepcopy(fold_results)), axis=0)
        #print("results: ", results)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model
        save_ensembles(model, model_out, iterator, str(iterator))

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
            #print("x and y: ", x, y)
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
                print("y, pred,", i.item(), j.item(), i == j)
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

    print("TP FP FN", true_pos, false_pos, false_neg)
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
    model_predictions = []
    #fold_prediction_sets = []

    #for fold in testing_data:
    #    print(fold)
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
        truths = []

        with torch.no_grad():
            for x, y in testing_data:#.dataset:
                x = x.to(device=LocalResnet.my_device)
                y = y.to(device=LocalResnet.my_device)
                truths.append(y.item())
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

        model_predictions.append(prediction_array)
    #fold_prediction_sets.append(model_predictions)

    voted_predictions = []
    confidence_predictions = []

    #length should be equal to number of image instances, fold_prediction_sets is rows of each models prediction for each image
    #Chris is 0, claire is 1
    print("model predictions and num instances the same value: ", len(model_predictions[0]), num_samples)

    for model_iter, model in enumerate(model_predictions):
        chris_votes = 0
        claire_votes = 0
        for iter, model_prediction in enumerate(model):
            ## iterate through each models prediction and tally the votes
            print("iter before: ", iter)
            if model_prediction == 0:
                chris_votes += 1
            else:
                claire_votes +=1

        if chris_votes > claire_votes:
            voted_predictions.append(0)
        else:
            voted_predictions.append(1)

        confidence_predictions.append([chris_votes, claire_votes])
    
    voting_confidences = []
    num_correct = 0
    num_chris = 0
    num_claire = 0
    num_correct_claire = 0
    num_correct_chris = 0

    # Calculate precision and recall
    true_pos = 0
    false_pos = 0
    false_neg = 0
    print("lengths: ", len(truths), len(voted_predictions), len(confidence_predictions[0]), len(confidence_predictions[1]), len(confidence_predictions))
    zipped = zip(truths, voted_predictions)
    for iter, (i, j) in enumerate(zipped):
        print(" i and j : ", i, j)
        if i == j:
            num_correct += 1
            if i == 0:
                num_correct_chris += 1
                voting_confidences.append(confidence_predictions[iter][0])
            else:
                num_correct_claire += 1
                voting_confidences.append(confidence_predictions[iter][1])
                true_pos += 1

        if i == 0:
            num_chris += 1
        else:
            num_claire += 1

        if i != j:
            if i == confidence_predictions[iter][0]:
                voting_confidences.append(confidence_predictions[iter][0])
            else:
                voting_confidences.append(confidence_predictions[iter][1])
            if j == 0:
                false_neg += 1
            elif j == 1:
                false_pos += 1

    print("voting confidence?", voting_confidences)

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


    total_confidence = sum(voting_confidences) / len(voting_confidences)
    total_accuracy = num_correct / num_samples * 100

    return [total_accuracy, total_confidence, precision, recall, f1_score]



#Function to save resnets
def save_ensembles(model, model_path, ensemble_count, fold_count):
    # Saving the model
    os.makedirs(model_path + 'ensemble_model_' + str(ensemble_count) + '/', exist_ok=True)
    save_path = model_path + 'model_fold_' + str(fold_count) + '.pth'
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
    epoch = 2
    target = Lambda( lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    ensemble_models =  create_ensemble_resnets(n, model_path='./Models/GEI_SpecialSilhouettes/model_fold_2.pth') # <- CHANGE EVERY MODEL!!

    #Split the few shot data
    #1 fold for each of the ensemble models to train individually
    few_shot_training, few_shot_testing = split_data_n_folds(num_folds=n,
                                                            sourceTransform=ToTensor(),
                                                            targetTransform=target,
                                                            sizes='./Instance_Counts/FewShot/Normal/GEI.csv', # <- change this between GEI or FFGEI/HOGFFGEI and graphcut
                                                            batch_size=batch_size,
                                                            FFGEI=False,
                                                            data_path='./Images/GEI/FewShot/SpecialSilhouettes', # <- Change this per experiment
                                                            label_path='./labels/FewShot/labels.csv') # <- Change this for Graphcuts

    #Currently training is done in the creation, split into the train, single model
    ensemble_models = train_ensemble_model(training_data = few_shot_training, testing_data = few_shot_testing, models = ensemble_models,
                                           epoch = epoch, batch_size = 1,
                                           results_out ='./Results/Ensemble/GEI_SpecialSilhouettes/', # <- Change this per experiment
                                           model_out = './Models/Ensemble/GEI_SpecialSilhouettes/' ) # <- Change this per experiment
        
    #Evaluate ensemble resnets and conduct voting for classification
    evaluation_results = evaluate_ensemble(ensemble_models, few_shot_testing[0])

    empty_models = create_ensemble_resnets(n, None, empty = True)
    #Evaluate standard models with no pre-trained weights as a control group
    control_results = evaluate_ensemble(empty_models, few_shot_testing[0])

    print("control results: ")
    #print(control_results)
    #Save the control and evaluation results (together maybe?) after that clean the code and we are all done :)
    #combined_results = np.concatenate((control_results, evaluation_results, axis=0)
    control_frame = pd.DataFrame([control_results])
    eval_frame = pd.DataFrame([evaluation_results])

    #Control only needs to be recorded once per GEI method (GEI, HOGFFGEI, FFGEI)
    append_average_metrics(control_results, control_frame, results_out = './Results/Few_Shot/Control/FFGEI/') # <- Change this per between GEI, FFGEI and HOGFFGEI

    append_average_metrics(evaluation_results, eval_frame, results_out = './Results/Few_Shot/Normal/FFGEI/SpecialSilhouettes/') # <- Change this per experiment

    #'./Results/Few_Shot/Normal/GEI_results.csv'
    #'./Results/Few_Shot/Normal/HOGFFGEI_results.csv'
    #'./Results/Few_Shot/Normal/FFGEI_results.csv'

    #GEI
    #   Special Silhouette - done
    #   Mask
    #   Graphcut

    #FFGEI
    #   Special Silhouette
    #   Mask
    #   Graphcut

    #HOGFFGEI
    #   Special Silhouette
    #   Mask

#Not appending averages: only place used is where a single example is used. for GEI evaluation maybe need to use multiple folds
def append_average_metrics(array, frame, results_out):
    means = []
    stds = []
    # Remove column labels
    tmp = frame#.iloc[1:, :]
    #print("temp array: ")
    #print(tmp)

    for column in tmp:
        stds.append(np.std(tmp[column]))
        means.append(tmp[column].mean())

    print("array currently", array)

    print("means and std dev:", means, stds)
    means[0] = 'Means'
    stds[0] = 'St.Devs'
    array.append(means)
    array.append(stds)

    print("array now: ", array)
    frame = pd.DataFrame([array])
    #print("array as a dataframe: ")
    print(frame.head())
    # Save as CSV all results
    os.makedirs(results_out, exist_ok=True)
    frame.to_csv(results_out + "results.csv")