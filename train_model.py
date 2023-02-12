#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets


import argparse
import os, json, sys


#TODO: Import dependencies for Debugging andd Profiling
import smdebug.pytorch as smd
from smdebug.profiler.utils import str2bool


def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # model = models.__dict__(args.arch)(pretrained=True)

    model = models.resnet18(pretrained=True)
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
            ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
            ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
            ]),
    }
    # create training, validation and test datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    # create training, validation and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

    return dataloaders_dict



def main(args):

    ## device agnostic
    device = 'cuda' if args.gpu == 1 and torch.cuda.is_available() else 'cpu'

    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model.to(device) ## move model to device, GPU if avalaible


    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameter(), lr=args.lr)



    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    dataloader = create_data_loaders(args.data_dir, args.batch_size)
    model=train(model, dataloader['train'], loss_criterion, optimizer)



    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, dataloader['test'], loss_criterion)



    '''
    TODO: Save the trained model
    '''
    torch.save(model, args.model_dir)



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--arch', type=str, default='resnet18', help='Load a pre-trained model (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=5, help='Number epochs for training (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=64, help='Enter number of train batch size (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Enter number of test batch size (default: 32)')
    parser.add_argument('--gpu', type=str2bool, default=True, help='Enable GPU acceleration for training (default: True)')

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args=parser.parse_args()

    main(args)
