import os

from colorization_model_classification import colorization_model_zhang
#from initialized_weights_colorization import colorization_model_zhang
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
    ab_channels = np.stack([a_values,b_values],axis=3)
    f = lambda p: soft_encoding(points,p,k)
    shape = ab_channels.shape

    newshape = list(shape[:-1])+[buckets]
    se = np.zeros(newshape)
    for i in range(shape[0]):
        for x in range(shape[1]):
            for y in range(shape[2]):
                se[i,x,y,:] = f(ab_channels[i,x,y,:])
    return se

def weighting(freq,num_bins):
    lam = 0.5
    p = gaussian_filter1d(freq,sigma=5)
    w = ((1-lam)*p+(lam/(num_bins)))**(-1)
    w_norm = w/w.sum()
    return w_norm

def create_histogram(dataloader):
    num_bins = buckets
    bins = np.arange(num_bins+1)
    running_total = np.zeros(num_bins).astype('float')

    for inputs,_ in dataloader:

        indexes = convert_one_hot(inputs)
        running_total += np.histogram(indexes, bins=bins, density=False)[0]
    freq = running_total / running_total.sum()
    return weighting(freq,num_bins)

def post_processing_one_hot(orig,output,criterion,histogram):
    lab_one_hot = torch.tensor(convert_one_hot(orig.cpu())).to(device)
    lab_one_hot = swap(lab_one_hot)
    return criterion(output,lab_one_hot)

def create_all_points(width,block_size):
    return np.array([np.array([x*block_size,y*block_size]) for x in range(-width,width+1) for y in range(-width,width+1)])

width = 4
block_size =15
points = create_all_points(width=width,block_size=block_size)
k = 5
sigma = 5
buckets = (width*2+1)**2

def create_loss(histogram):
    return lambda outputs,expected: custom_cross_entropy(outputs,expected,histogram)
def custom_cross_entropy(output,expected,histogram):
    shape = output.size()
    expected = iswap(expected)
    output = iswap(output)
    return -torch.sum(histogram*expected*torch.log(output))/(shape[0]*shape[1]*shape[2])

def indexToValue(index,block_size):
    # index[Batch_size,size,size,1]
    length = (width*2+1)
    a_index = index//length
    b_index = (index)%length

    a = (a_index-width)*block_size
    b = (b_index-width)*block_size
    return np.stack([a,b],axis=3)

def create_all_points(width,block_size):
    return np.array([np.array([x*block_size,y*block_size]) for x in range(-width,width) for y in range(-width,width)])

def soft_encoding(points,point,k):
    distances = np.linalg.norm(points-point,axis=1)
    indexes = sorted(np.arange(points.shape[0]),key = lambda a: distances[a])[:k]
    #weights = (1/(np.sqrt(np.pi*2)*sigma))*np.exp(-(np.arange(k)**2)/(2*sigma**2))
    weights = np.array([1,0.5,0.25,0.125,0.0625])
    zeros = np.zeros(points.shape[0])
    zeros[indexes] = weights
    return  zeros/zeros.sum()


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

    #image_datasets = getRandomDataSets(data_transforms)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_transforms.keys()}

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
            #break
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
            c = convert_one_hot(inputs)

            #c = swap(c)
            inputs = iswap(inputs)
            outputs = iswap(outputs)
            gray = iswap(gray)
            #f, ax = plt.subplots(inputs.shape[0],4) 
            f, ax = plt.subplots(2,6) 
            ax[0][0].set_title("Original")
            ax[0][1].set_title("Grayscale")
            ax[0][2].set_title("Best Chrominance")
            ax[0][3].set_title("Chrominance")
            ax[0][4].set_title("Best coloring")
            ax[0][5].set_title("Recolored")

            pred_indexes = np.argmax(outputs,axis=3)

            best_ab = one_hot_to_image(c)

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
                remove_axis(ax[i][5])
                ax[i][1].imshow(gray[i])

                bc = best_ab[i]
                full_bc = np.stack([np.full(lab[:,:,0].shape,50),bc[:,:,0],bc[:,:,1]],axis=2).astype(float)
                ax[i][2].imshow(color.lab2rgb(full_bc))
                
                chromanince = np.stack([np.full(lab[:,:,0].shape,50),ab_channels[:,:,0],ab_channels[:,:,1]],axis=2).astype(float)
                rgb_chrome = color.lab2rgb(chromanince)
                ax[i][3].imshow(rgb_chrome)

                ax[i][4].imshow(color.lab2rgb(np.stack([lab[:,:,0],bc[:,:,0],bc[:,:,1]],axis=2)))

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
    optimizer = optim.Adam(params_to_update, lr=0.0003)
    return optimizer

def get_loss(histogram):
    # Create an instance of the loss function
    #criterion = nn.CrossEntropyLoss(weight = histogram)
    #criterion = nn.CrossEntropyLoss()
    #criterion = custom_cross_entropy
    return create_loss(histogram)




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
    save_file = "soft_encoding_k_5_xpo_histogram"
    resume_from = None 

    model_stats = {}
    model_stats["Batch Size"] = batch_size
    model_stats["num_epochs"] = num_epochs
    model_stats["model_name"] = model_name
    os.makedirs('outputs/'+save_file, exist_ok=True)
    os.makedirs("plots/" + save_file,exist_ok=True)

    input_size = 224
    classification_feature_size = 128#classification_model
    colorization_model = colorization_model_zhang(buckets)
    dataloaders = get_dataloaders(device, input_size, batch_size, shuffle_datasets)

    histogram = torch.Tensor(create_histogram(dataloaders["train"]))
    #histogram = None
    criterion = get_loss(histogram)

    colorization_model = colorization_model.to(device)

    optimizer = make_optimizer(colorization_model)

    trained_model, validation_history = train_model(histogram=histogram,device=device,model_name=save_file, colorization_model=colorization_model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
           save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=num_epochs)

