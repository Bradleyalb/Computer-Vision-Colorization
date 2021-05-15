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

data_dir = "./imagenet_data"


MAX_ITER = float("inf")

def initialize_classification_model(model_name, num_classes, pretrained = True, resume_from = None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # The model (nn.Module) to return
    model_ft = None
    # The input image is expected to be (input_size, input_size)
    input_size = 224
    
    # By default, all parameters will be trained (useful when you're starting from scratch)
    # Within this function you can set .requires_grad = False for various parameters, if you
    # don't want to learn them
    use_pretrained = True
    if model_name == "resnet":
        """ Resnet18
        """
        resnet = models.resnet18(pretrained=use_pretrained)
        model_ft = nn.Sequential(*list(resnet.children())[0:6])
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

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        raise Exception("Invalid model name!")
    
    #Don't train any of the pretrained models
    for param in model_ft.parameters():
        param.requires_grad = False

    if resume_from is not None:
        print("Loading weights from %s" % resume_from)
        model_ft.load_state_dict(torch.load("weights/"+resume_from))

    return model_ft, input_size

def initialize_colorization_model(k,classification_feature_size):
    '''
    k: The number of onehot vector classifications
    input_size: The shape the model expects for input.
    '''
    return nn.Sequential(     
      nn.Conv2d(classification_feature_size, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, number_lab_classes, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2),
      nn.Softmax2d()
    )

max_value = 70
min_value = -70
lab_range = max_value-min_value
block_size = 7
number_lab_classes = (lab_range//block_size+1)**2

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

def convert_one_hot(image):
    image = np.swapaxes(image,1,3)
    image = np.swapaxes(image,1,2)
    lab = color.rgb2lab(image)
    a_values = lab[:,:,:,1]
    b_values = lab[:,:,:,2]

    indexes = quantizeLAB(a_values,b_values,block_size)

    # one_hot_image = to_one_hot(indexes, number_lab_classes)
    # one_hot_image = swap(one_hot_image)
    return indexes


def to_one_hot(indexes,n_classes):
    one_hot = np.zeros((indexes.shape[0],indexes.shape[1], indexes.shape[2], n_classes))
    for im_index in range(indexes.shape[0]):
        cur_image = indexes[im_index]
        for i, unique_value in enumerate(np.unique(cur_image)):
            one_hot[im_index,:, :, i][cur_image == unique_value] = 1
    return one_hot

def post_processing_one_hot(orig,output,criterion):
    lab_one_hot = convert_one_hot(orig)    
    return criterion(output, torch.tensor(lab_one_hot).long())

def post_processing(orig,output,criterion):
    #one_hot_orig = convert_one_hot(orig)
    image = np.swapaxes(orig,1,3)
    image = np.swapaxes(image,1,2)
    lab = color.rgb2lab(image)
    print(lab.shape)
    lab = np.swapaxes(lab,1,2)
    lab = np.swapaxes(lab,1,3)


    return criterion(torch.tensor(lab[:,1:,:,:]).float(),output)

def loss(one_hot_orig,output):
    return nn.MSELoss(one_hot_orig.float(),output.float())

def indexToValue(index,block_size):
    width = (max_value-min_value)/block_size
    a_index = index//width
    b_index = (index)%width

    a = a_index*block_size+min_value
    b = b_index*block_size+min_value
    return np.stack([a,b],axis=3)


def one_hot_to_image(one_hot_image):
    indexes = np.argmax(one_hot_image,axis=3)
    return indexToValue(indexes,block_size)


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
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'example': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()#,
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # if random_perspective:
    #     data_transforms['train'].insert(0,transforms.RandomPerspective())
    # if mirror_data:
    #     data_transforms['train'].insert(0,transforms.RandomHorizontalFlip(p=0.5))
    # if random_jitter:
    #     data_transforms['train'].insert(0,transforms.ColorJitter(brightness=0.3))
    data_transforms['train'] = transforms.Compose(data_transforms['train'])
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_transforms.keys()}
    # Create training and validation dataloaders
    # Never shuffle the test set
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=4) for x in data_transforms.keys()}
    return dataloaders_dict

def forwards(inputs, classification_model, colorization_model):
    gray = transforms.Grayscale(num_output_channels=3)(inputs)
    mid_level_features = classification_model(gray)
    return colorization_model(mid_level_features)

def train_model(device,model_name ,classification_model,colorization_model, dataloaders, criterion, optimizer, save_dir = None, save_all_epochs=False, num_epochs=25):
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
            num_iter = 0
            for inputs, _ in tqdm(dataloaders[phase]):
                if num_iter > MAX_ITER:
                  break
                num_iter += 1
                inputs = inputs.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = forwards(inputs,classification_model,colorization_model)

                    loss = post_processing_one_hot(inputs,outputs,criterion)

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
            gray = transforms.Grayscale(num_output_channels=3)(inputs)
            outputs = forwards(inputs, classification_model, colorization_model)
            outputs = outputs.cpu().detach().numpy()
            inputs = inputs.cpu().detach().numpy()
            inputs = np.swapaxes(inputs,1,3)
            inputs = np.swapaxes(inputs,1,2)
            outputs = np.swapaxes(outputs,1,3)
            outputs = np.swapaxes(outputs,1,2)
            gray = np.swapaxes(gray,1,3)
            gray = np.swapaxes(gray,1,2)
            f, ax = plt.subplots(inputs.shape[0],4) 
            ax[0][0].set_title("Original")
            ax[0][1].set_title("Grayscale")
            ax[0][2].set_title("Chromanince")
            ax[0][3].set_title("Recolored")
            ab_channels_batch = one_hot_to_image(outputs)
            for i in range(inputs.shape[0]):
                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                ab_channels = ab_channels_batch[i]
                inp = inputs[i]
                ax[i][0].imshow(inp)
                remove_axis(ax[i][0])
                remove_axis(ax[i][1])
                remove_axis(ax[i][2])
                remove_axis(ax[i][3])
                ax[i][1].imshow(gray[i])

                lab = color.rgb2lab(inp)
                chromanince = np.stack([np.full(lab[:,:,0].shape,50),ab_channels[:,:,0],ab_channels[:,:,1]],axis=2)
                ax[i][2].imshow(color.lab2rgb(chromanince))
                
                full_output = np.stack([lab[:,:,0],ab_channels[:,:,0],ab_channels[:,:,1]],axis=2)
                ax[i][3].imshow(color.lab2rgb(full_output))
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
    optimizer = optim.Adam(params_to_update, lr=0.002)
    return optimizer

def get_loss():
    # Create an instance of the loss function
    criterion = nn.CrossEntropyLoss()
    return criterion


def evaluate(classification_model,colorization_model, dataloader, criterio):
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
    for inputs, _ in tqdm(dataloader):
        num_iters += 1
        if num_iters > MAX_ITER:
          break
        inputs = inputs.to(device)
        #labels = labels.to(device)
        #tiled_labels = torch.stack([labels.data for i in range(k)], dim=1) 
        # Makes this to calculate "top 5 prediction is correct"
        # [[label1 label1 label1 label1 label1], [label2 label2 label2 label label2]]

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = forwards(inputs,classification_model,colorization_model)

            loss = post_processing(inputs,outputs)

            # torch.topk outputs the maximum values, and their indices
            # Since the input is batched, we take the max along axis 1
            # (the meaningful outputs)

        running_loss += loss.item() * inputs.size(0)


    epoch_loss = float(running_loss / len(dataloader.dataset))


    
    # Return everything
    return epoch_loss









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
    #scheduler = pd.read_excel(schedule_name + ".xls")

    num_classes = 100
    shuffle_datasets = True
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    data_dir = './small_test_data'
    
    save_all_epochs = False

    #print(scheduler)
    model_values = {}
    for index,row in zip(range(1),range(1)):#scheduler.iterrows():
      start_time = time.perf_counter()
      
      model_name = "resnet" #row["Model Name"]
      batch_size = 8 #row["Batch Size"]
      num_epochs = 20 #row["Num Epochs"]
      save_file = "onehot_cross_entropy_block_size_7_adam"#row["Network Name"]
      resume_from = None #row["Resume From"]
      k = 25

      model_stats = {}
      model_stats["Batch Size"] = batch_size
      model_stats["num_epochs"] = num_epochs
      model_stats["model_name"] = model_name
      os.makedirs('outputs/'+save_file, exist_ok=True)
      os.makedirs("plots/" + save_file,exist_ok=True)

      classification_model, input_size = initialize_classification_model(model_name = model_name, num_classes = num_classes, resume_from = resume_from)
      classification_feature_size = 128#classification_model
      colorization_model = initialize_colorization_model(k,classification_feature_size)
      dataloaders = get_dataloaders(device, input_size, batch_size, shuffle_datasets)
      criterion = get_loss()
      classification_model = classification_model.to(device)
      colorization_model = colorization_model.to(device)

      optimizer = make_optimizer(colorization_model)

      trained_model, validation_history = train_model(device=device,model_name=save_file, classification_model=classification_model, colorization_model=colorization_model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
               save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=num_epochs)
      
      end_time = time.perf_counter()
      duration = end_time-start_time

      

      generate_validation_labels = True
      #val_loss, val_top1, val_top5, val_labels = evaluate(model, dataloaders['val'], criterion, is_labelled = True, generate_labels = generate_validation_labels, k = 5)

      #_, _, _, test_labels = evaluate(model, dataloaders['test'], criterion, is_labelled = False, generate_labels = True, k = 5)
      

      model_values[save_file] = model_stats

