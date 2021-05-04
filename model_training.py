import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
# You might not have tqdm, which gives you nice progress bars
from tqdm import tqdm
import copy

import pandas as pd

from skimage import io, color
from skimage import data
from skimage.color import rgb2lab, lab2lch, lab2rgb

data_dir = "./data"


MAX_ITER = float("inf")

def initialize_model(model_name, num_classes, resume_from = None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # The model (nn.Module) to return
    model_ft = None
    # The input image is expected to be (input_size, input_size)
    input_size = 0
    
    # You may NOT use pretrained models!! 
    use_pretrained = False
    
    # By default, all parameters will be trained (useful when you're starting from scratch)
    # Within this function you can set .requires_grad = False for various parameters, if you
    # don't want to learn them

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
      model_ft = models.resnet50(pretrained=use_pretrained)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs,num_classes)
      input_size = 224

    elif model_name == "resnet101":
      model_ft = models.resnet101(pretrained=use_pretrained)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs,num_classes)
      input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    else:
        raise Exception("Invalid model name!")
    
    if resume_from is not None:
        print("Loading weights from %s" % resume_from)
        model_ft.load_state_dict(torch.load("weights/"+resume_from))
    
    return model_ft, input_size

def postProcessing(orig,output):
    lab = LabColor(0.903, 16.296, -2.22)
    full_lab = torch.stack(orig,output)
    img_rgb = lab2rgb(full_lab)
    
    loss(img_rgb)

def loss(img_rgb,expected):
    nn.MSELoss(img_rgb,expected)


def get_dataloaders(device,input_size, batch_size, shuffle = True, mirror_data= True,random_flip=False, random_jitter=False,random_crop=False,random_perspective=False):
    # How to transform the image when you are loading them.
    # you'll likely want to mess with the transforms on the training set.
    
    # For now, we resize/crop the image to the correct input size for our network,
    # then convert it to a [C,H,W] tensor, then normalize it to values with a given mean/stdev. These normalization constants
    # are derived from aggregating lots of data and happen to produce better results.
    data_transforms = {
        'train': [
            transforms.functional.to_grayscale(),
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size) if random_crop else transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ],
        'val': transforms.Compose([
            transforms.functional.to_grayscale(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.functional.to_grayscale(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    if random_perspective:
        data_transforms['train'].insert(0,transforms.RandomPerspective())
    if mirror_data:
        data_transforms['train'].insert(0,transforms.RandomHorizontalFlip(p=0.5))
    if random_jitter:
        data_transforms['train'].insert(0,transforms.ColorJitter(brightness=0.3))
    data_transforms['train'] = transforms.Compose(data_transforms['train'])
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_transforms.keys()}
    # Create training and validation dataloaders
    # Never shuffle the test set
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=4) for x in data_transforms.keys()}
    return dataloaders_dict

def train_model(max_train_time,device,model_name ,model, dataloaders, criterion, optimizer, save_dir = None, save_all_epochs=False, num_epochs=25):
    '''
    model: The NN to train
    dataloaders: A dictionary containing at least the keys 
                 'train','val' that maps to Pytorch data loaders for the dataset
    criterion: The Loss function
    optimizer: The algorithm to update weights 
               (Variations on gradient descent)
    num_epochs: How many epochs to train for
    save_dir: Where to save the best model weights that are found, 
              as they are found. Will save to save_dir/weights_best.pt
              Using None will not write anything to disk
    save_all_epochs: Whether to save weights for ALL epochs, not just the best
                     validation error epoch. Will save to save_dir/weights_e{#}.pt
    '''
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    total_loss = []
    total_acc = []
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        if time.perf_counter() - start_time > max_train_time:
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # TQDM has nice progress bars
            num_iter = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                if num_iter > MAX_ITER:
                  break
                num_iter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # torch.max outputs the maximum value, and its index
                    # Since the input is batched, we take the max along axis 1
                    # (the meaningful outputs)
                    _, preds = torch.max(outputs, 1)

                    # backprop + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            total_loss.append(epoch_loss)
            total_acc.append(epoch_loss)

            val_loss, val_top1, val_top5, val_labels = evaluate(model, dataloaders['val'], criterion, is_labelled = True, generate_labels = generate_validation_labels, k = 5)
            train_loss, train_top1, train_top5, train_labels = evaluate(model, dataloaders['val'], criterion, is_labelled = True, generate_labels = generate_validation_labels, k = 5)

            if phase == "train":
                train_acc_history.append(train_top5)

                plt.plot(val_acc_history)
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(model_name + " Accuracy")
                plt.savefig("plots/" + model_name + ' Accuracy.png')
                plt.clf()
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(save_dir, model_name + '_best.pt'))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

                plt.plot(val_acc_history)
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(model_name + " Accuracy")
                plt.savefig("plots/" + model_name + ' Accuracy.png')
                plt.clf()
            if save_all_epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, f'weights_{epoch}.pt'))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    # save and load best model weights
    torch.save(best_model_wts, os.path.join(save_dir, model_name + '_best.pt'))
    model.load_state_dict(best_model_wts)
    
    return model, val_acc_history


def make_optimizer(model):
    # Get all the parameters
    params_to_update = model.parameters()
    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # Use SGD
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer

def get_loss():
    # Create an instance of the loss function
    criterion = nn.MSELoss()
    return criterion


def evaluate(model, dataloader, criterion, is_labelled = False, generate_labels = True, k = 5):
    # If is_labelled, we want to compute loss, top-1 accuracy and top-5 accuracy
    # If generate_labels, we want to output the actual labels
    # Set the model to evaluate mode
    model.eval()
    running_loss = 0
    running_top1_correct = 0
    running_top5_correct = 0
    predicted_labels = []
    

    # Iterate over data.
    # TQDM has nice progress bars
    num_iters = 0
    for inputs, labels in tqdm(dataloader):
        num_iters += 1
        if num_iters > MAX_ITER:
          break
        inputs = inputs.to(device)
        labels = labels.to(device)
        tiled_labels = torch.stack([labels.data for i in range(k)], dim=1) 
        # Makes this to calculate "top 5 prediction is correct"
        # [[label1 label1 label1 label1 label1], [label2 label2 label2 label label2]]

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            if is_labelled:
                loss = criterion(outputs, labels)

            # torch.topk outputs the maximum values, and their indices
            # Since the input is batched, we take the max along axis 1
            # (the meaningful outputs)
            _, preds = torch.topk(outputs, k=5, dim=1)
            if generate_labels:
                # We want to store these results
                nparr = preds.cpu().detach().numpy()
                predicted_labels.extend([list(nparr[i]) for i in range(len(nparr))])

        if is_labelled:
            # statistics
            running_loss += loss.item() * inputs.size(0)
            # Check only the first prediction
            running_top1_correct += torch.sum(preds[:, 0] == labels.data)
            # Check all 5 predictions
            running_top5_correct += torch.sum(preds == tiled_labels)
        else:
            pass

    # Only compute loss & accuracy if we have the labels
    if is_labelled:
        epoch_loss = float(running_loss / len(dataloader.dataset))
        epoch_top1_acc = float(running_top1_correct.double() / len(dataloader.dataset))
        epoch_top5_acc = float(running_top5_correct.double() / len(dataloader.dataset))
    else:
        epoch_loss = None
        epoch_top1_acc = None
        epoch_top5_acc = None
    
    # Return everything
    return epoch_loss, epoch_top1_acc, epoch_top5_acc, predicted_labels









if __name__ == '__main__':
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()
     

    # dd/mm/YY H:M:S
    dt_string = now.strftime("Model values %d-%m-%Y %H!%M!%S")
    print("date and time =", dt_string)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only")
    # Detect if we have a GPU available
    schedule_name = "resnet_augmentations"
    scheduler = pd.read_excel(schedule_name + ".xls")
    num_classes = 100
    shuffle_datasets = True
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    data_dir = './data'
    
    save_all_epochs = False

    print(scheduler)
    model_values = {}
    for index,row in scheduler.iterrows():
      start_time = time.perf_counter()
      
      model_name = row["Model Name"]
      batch_size = row["Batch Size"]
      num_epochs = row["Num Epochs"]
      save_file = row["Network Name"]
      MAX_TRAIN_TIME = row["Max Train Time"]
      random_flip = row["Random Flip"]
      random_jitter = row["Random Jitter"]
      resume_from = row["Resume From"]
      random_perspective =row["Random Perspective"]
      random_crop = row["Random Crop"]


      model_stats = {}
      model_stats["Batch Size"] = batch_size
      model_stats["num_epochs"] = num_epochs
      model_stats["model_name"] = model_name

      model, input_size = initialize_model(model_name = model_name, num_classes = num_classes, resume_from = resume_from)
      dataloaders = get_dataloaders(device,input_size, batch_size, shuffle_datasets, random_flip = random_flip, random_jitter=random_jitter, random_perspective=random_perspective,random_crop=random_crop)
      criterion = get_loss()
      model = model.to(device)

      optimizer = make_optimizer(model)

      trained_model, validation_history = train_model(max_train_time=MAX_TRAIN_TIME,device=device,model_name=save_file, model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
               save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=num_epochs)
      
      end_time = time.perf_counter()
      duration = end_time-start_time

      

      generate_validation_labels = True
      val_loss, val_top1, val_top5, val_labels = evaluate(model, dataloaders['val'], criterion, is_labelled = True, generate_labels = generate_validation_labels, k = 5)

      _, _, _, test_labels = evaluate(model, dataloaders['test'], criterion, is_labelled = False, generate_labels = True, k = 5)


      model_stats["Loss"] = val_loss
      model_stats["Top1 accuracies"] = val_top1
      model_stats["Top5 Accuracies"] = val_top5
      model_stats["Duration"] = duration
      model_stats["MAX Train Time"] = MAX_TRAIN_TIME
      model_stats["random_flip"] = random_flip
      model_stats["random_jitter"] = random_jitter
      model_stats["resume_from"] = resume_from
      

      model_values[save_file] = model_stats

    df = pd.DataFrame.from_dict(model_values, orient='index') # convert dict to dataframe

    
    df.to_csv(schedule_name + '.csv') # write dataframe to file





