import os

from colorization_model_classification import colorization_model_zhang
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

from scipy.ndimage import gaussian_filter1d

max_value = 80
min_value = -80
lab_range = max_value-min_value
block_size = 12
number_lab_classes = (lab_range//block_size+1)**2

def weighting(freq,num_bins):
    lam = 0.5
    p = gaussian_filter1d(freq,sigma=5)
    w = ((1-lam)*p+(lam/(num_bins)))**(-1)
    w_norm = w/w.sum()
    return w_norm

def create_histogram(dataloader):
    num_bins = number_lab_classes
    bins = np.arange(num_bins+1)
    running_total = np.zeros(num_bins).astype('float')

    for inputs,_ in dataloader:

        indexes = convert_one_hot(inputs)
        running_total += np.histogram(indexes, bins=bins, density=False)[0]
    freq = running_total / running_total.sum()
    return weighting(freq,num_bins)

def quantizeLAB(a,b,block_size):
    width = (max_value-min_value)/block_size
    a = np.clip(a,min_value,max_value)
    b = np.clip(b,min_value,max_value)
    a_index = (a-min_value)//block_size
    b_index = (b-min_value)//block_size
    index = a_index*width + b_index
    return index

def swap(image):
    image = np.swapaxes(image,1,2)
    return np.swapaxes(image,1,3)

def iswap(image):
    '''
    Switching from Pytorch to plt
    '''
    image = np.swapaxes(image,1,3)
    return np.swapaxes(image,1,2)
def convert_one_hot(image):
    image = iswap(image)
    lab = color.rgb2lab(image)
    a_values = lab[:,:,:,1]
    b_values = lab[:,:,:,2]

    indexes = quantizeLAB(a_values,b_values,block_size)
    # one_hot_image = to_one_hot(indexes, number_lab_classes)
    # one_hot_image = swap(one_hot_image)
    return indexes

def post_processing_one_hot(orig,output,criterion,histogram):
    lab_one_hot = torch.tensor(convert_one_hot(orig.cpu())).to(device)
    return criterion(output,lab_one_hot.long())

def indexToValue(index,block_size):
    # index[Batch_size,size,size,1]
    width = (max_value-min_value)/block_size
    a_index = index//width
    b_index = (index)%width

    a = a_index*block_size+min_value
    b = b_index*block_size+min_value
    return np.stack([a,b],axis=3)


def index_to_image(indexes):
    return indexToValue(indexes,block_size)
def one_hot_to_image(one_hot_image):
    indexes = np.argmax(one_hot_image,axis=3)
    return index_to_image(indexes)


def get_dataloaders(device,input_size, batch_size, shuffle = True, mirror_data= True,random_flip=False, random_jitter=False,random_crop=False,random_perspective=False):
    # How to transform the image when you are loading them.
    # you'll likely want to mess with the transforms on the training set.
    
    # For now, we resize/crop the image to the correct input size for our network,
    # then convert it to a [C,H,W] tensor, then normalize it to values with a given mean/stdev. These normalization constants
    # are derived from aggregating lots of data and happen to produce better results.
    data_transforms = {
        'train': [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()#,
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ],
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()#,
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'example': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()#,
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_transforms['train'] = transforms.Compose(data_transforms['train'])

    image_datasets = getRandomDataSets(data_transforms)
    #image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_transforms.keys()}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=4) for x in data_transforms.keys()}
    return dataloaders_dict

def getRandomDataSets():
    train,val = random_split(datasets.ImageFolder("data",data_transforms["train"],[33743,7257],generator=torch.Generator().manual_seed(42)))
    image_datasets['train'] = train
    image_datasets['val'] = val
    image_datasets['example'] = datasets.ImageFolder('example', data_transforms['example'])
    return image_datasets

def forwards(inputs, colorization_model):
    gray = transforms.Grayscale(num_output_channels=1)(inputs)
    return colorization_model(gray)

def train_model(histogram,device,model_name ,colorization_model, dataloaders, criterion, optimizer, save_dir = None, save_all_epochs=False, num_epochs=25):
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

    best_model_wts = copy.deepcopy(colorization_model.state_dict())
    best_loss = float('inf')

    train_loss = []
    val_loss = []
    start_time = time.perf_counter()
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                colorization_model.train()  # Set model to training mode
            else:
                colorization_model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            # TQDM has nice progress bars
            for inputs, _ in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = forwards(inputs,colorization_model)
                    print("outpus",outputs.shape)
                    loss = post_processing_one_hot(inputs,outputs,criterion,histogram)

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
                #running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("epoch_loss",epoch_loss)

            #val_loss = evaluate(classification_model,colorization_model, dataloaders['val'], criterion, is_labelled = True, generate_labels = generate_validation_labels, k = 5)

            if phase == "train":
                train_loss.append(epoch_loss)
                plt.clf()
                plt.plot(train_loss)
                plt.xlabel("Epoch")
                plt.ylabel("Train Loss")
                plt.title(model_name + " Accuracy")
                plt.savefig("plots/" + model_name +'/'+ 'TrainAccuracy.png')
                plt.clf()
                pass
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(colorization_model.state_dict())
                torch.save(best_model_wts, os.path.join(save_dir, model_name + '_best.pt'))
            if phase == 'val':
                val_loss.append(epoch_loss)
                plt.clf()
                plt.plot(val_loss)
                plt.xlabel("Epoch")
                plt.ylabel("Val Loss")
                plt.title(model_name + " Accuracy")
                plt.savefig("plots/" + model_name +'/'+ 'ValAccuracy.png')
                plt.clf()
                pass
            if save_all_epochs:
                torch.save(colorization_model.state_dict(), os.path.join(save_dir, f'weights_{epoch}.pt'))
        print()
        for inputs, _ in tqdm(dataloaders['example']):
            #Create expected best chrominance
            gray = transforms.Grayscale(num_output_channels=3)(inputs)
            inputs = inputs.to(device)
            outputs = forwards(inputs,colorization_model)
            outputs = outputs.cpu().detach().numpy()
            inputs = inputs.cpu().detach().numpy()
            best_indexes = convert_one_hot(inputs)
            inputs = iswap(inputs)
            outputs = iswap(outputs)
            gray = iswap(gray)
            #f, ax = plt.subplots(inputs.shape[0],4) 
            f, ax = plt.subplots(2,6) 
            ax[0][0].set_title("Original")
            ax[0][1].set_title("Grayscale")
            ax[0][2].set_title("Best Chrominance")
            ax[0][3].set_title("Chrominance")
            ax[0][4].set_title("Correct mask")
            ax[0][5].set_title("Recolored")

            pred_indexes = np.argmax(outputs,axis=3)
            
            best_chromanince = index_to_image(best_indexes)
            ab_channels_batch = one_hot_to_image(outputs)

            for i in range(inputs.shape[0]):
                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                ab_channels = ab_channels_batch[i]
                inp = inputs[i]
                lab = color.rgb2lab(inp)
                ax[i][0].imshow(inp)
                remove_axis(ax[i][0])
                remove_axis(ax[i][1])
                remove_axis(ax[i][2])
                remove_axis(ax[i][3])
                remove_axis(ax[i][4])
                ax[i][1].imshow(gray[i])

                bc = best_chromanince[i]
                full_bc = np.stack([np.full(lab[:,:,0].shape,50),bc[:,:,0],bc[:,:,1]],axis=2)
                ax[i][2].imshow(color.lab2rgb(full_bc))
                
                chromanince = np.stack([np.full(lab[:,:,0].shape,50),ab_channels[:,:,0],ab_channels[:,:,1]],axis=2)
                ax[i][3].imshow(color.lab2rgb(chromanince))
                
                bi = best_indexes[i] 
                pi = pred_indexes[i]
                mask = np.where(bi==pi,1,0)
                ax[i][4].imshow(mask,'gray')

                full_output = np.stack([lab[:,:,0],ab_channels[:,:,0],ab_channels[:,:,1]],axis=2)
                ax[i][5].imshow(color.lab2rgb(full_output))
            plt.savefig('outputs/'+save_file + '/figure epoch number ' + str(epoch) +'.png')
            break





    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))


    # save and load best model weights
    torch.save(best_model_wts, os.path.join(save_dir, model_name + '_best.pt'))
    colorization_model.load_state_dict(best_model_wts)
    
    return colorization_model, val_acc_history



def remove_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
def make_optimizer(model):
    # Get all the parameters
    params_to_update = model.parameters()
    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # Use SGD
    optimizer = optim.Adam(params_to_update, lr=0.001)
    return optimizer

def get_loss(histogram):
    # Create an instance of the loss function
    criterion = nn.CrossEntropyLoss(weight = histogram)
    return criterion




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only")

    shuffle_datasets = True
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    #data_dir = "./fruit"
    data_dir = "./small_test_data"    
    save_all_epochs = False
      
    model_name = "resnet" 
    batch_size = 32 
    num_epochs = 50 
    save_file = "fc_lr_0.001_new_weights_small_zhangd"
    resume_from = None 

    model_stats = {}
    model_stats["Batch Size"] = batch_size
    model_stats["num_epochs"] = num_epochs
    model_stats["model_name"] = model_name
    os.makedirs('outputs/'+save_file, exist_ok=True)
    os.makedirs("plots/" + save_file,exist_ok=True)

    input_size = 224
    classification_feature_size = 128#classification_model
    colorization_model = colorization_model_zhang(number_lab_classes)
    dataloaders = get_dataloaders(device, input_size, batch_size, shuffle_datasets)

    histogram = torch.Tensor(create_histogram(dataloaders["train"]))
    criterion = get_loss(histogram)

    colorization_model = colorization_model.to(device)

    optimizer = make_optimizer(colorization_model)

    trained_model, validation_history = train_model(histogram=histogram,device=device,model_name=save_file, colorization_model=colorization_model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
           save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=num_epochs)

