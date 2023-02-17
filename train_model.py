#TODO: Import your dependencies.

import argparse
import json
import logging
import os
import sys

#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

import argparse
import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


#TODO: Import dependencies for Debugging andd Profiling

def test(model, valid_loader, criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss = 0
    running_samples = 0 
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            pred = model(inputs)
            loss = criterion(pred, labels)
            running_loss += loss.item()*inputs.size(0)
            pred = pred.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(labels.view_as(pred)).sum().item()
            
        
        
        
    test_loss = running_loss/len(valid_loader.dataset)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, accuracy, len(valid_loader.dataset), 100*accuracy/len(valid_loader.dataset)))
    
    
    

def train(model, train_loader, valid_loader, criterion, optimizer, args, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    
    for epoch in range(1, args.epoch+1):
        
        model.train()
        hook.set_mode(smd.modes.TRAIN)
        running_loss = 0
        #running_samples = 0 
        accuracy = 0
        batch_idx = 0
        for inputs, labels in train_loader:

            batch_idx += 1
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            running_loss += loss*inputs.size(0)
            pred = pred.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(labels.view_as(pred)).sum().item()
            #running_samples += data.size(0)
            loss.backward()
            optimizer.step()
           
            if batch_idx % 27 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]  Train Accuracy: {:.3f}% Train Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        100*accuracy/(batch_idx*len(inputs)), 
                        loss.item()
                    )
                )
                test(model, valid_loader, criterion, hook)

        
    return model
    
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1000),                              
                             nn.ReLU(),
                             nn.Linear(1000, 133),
                             nn.LogSoftmax(dim = 1))
    return model

def create_data_loaders(data_dir, batch_size=32):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    #Load the datasets with ImageFolder
    train_images = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_images = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_images = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_images, batch_size = 32, shuffle = True )
    valid_loader = torch.utils.data.DataLoader(validation_images, batch_size = 32)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size = 32)
    print('train loader size is: {} test loader size is : {} '.format(len(train_loader), len(valid_loader)))
    return (train_loader, valid_loader, test_loader)
    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    #hook = get_hook(create_if_not_exists=True)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.NLLLoss()
    hook.register_loss(loss_criterion)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, batch_size=args.batch_size)
    
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, args, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, valid_loader, loss_criterion, hook)
    
    '''
    TODO: Save the trained model
    '''
    
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    print('model saved at: ', args.model_dir)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.002329397097607175)
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args=parser.parse_args()
    
    
    main(args)
