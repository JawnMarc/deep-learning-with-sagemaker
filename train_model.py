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


def test(model, dataloader, citreon, device, hook):
    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(input)
            loss = citreon(output, labels)
            _, preds = torch.max(output, 1)

            test_loss += loss.item() * input.size(0)
            test_acc += torch.sum(preds == labels.data)

        epoch_loss = test_loss/len(dataloader['test'].dataset)
        accuracy = test_acc.double()/len(dataloader['test'].dataset)

        print(f'Test: \tLoss: {epoch_loss} \t Test Acc: {accuracy}')





def train(model, dataloader, criterion, optimizer,num_epochs, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)

            running_loss = 0
            correct = 0

            for inputs, labels in dataloader[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # reset gradient to zero
                optimizer.zero_grad()

                # forward
                # with torch.set_grad_enabled(phase=='train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward and optimize if training mode
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # getting statistics
                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss/len(dataloader[phase].dataset)
            epoch_acc = correct.double()/len(dataloader[phase].dataset)

            print(f'{phase}: \tLoss: {epoch_loss} \tAcc: {epoch_acc}')



def net(model_name):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''

    num_classes = 133 # number of classes in dataset
    # load a pre-trained network
    model = models.__dict__[model_name](pretrained=True)
    
    # load resnet18
    if model_name == 'resnet18':
        input_feat = model.fc.in_features
        model.fc = nn.Linear(input_feat, num_classes)

    elif model_name == 'vgg13':
        input_feat = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(input_feat, num_classes)

    # model = models.resnet18(pretrained=True)
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
        'valid': transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
            ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
            ]),
    }

    # create training, validation and test datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data, x), data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    # create training, validation and test dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
        for x in ['train', 'valid', 'test']}

    return dataloaders_dict



def main(args):
    ## device agnostic
    device = 'cuda' if args.gpu == True and torch.cuda.is_available() else 'cpu'

    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net(args.arch)
    model.to(device) ## move model to device, GPU if avalaible


    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    # create hook for debugging   
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(optimizer)

    '''
    create_data_loaders returns a dictionery. Key names: 'train', 'val', 'test'
    '''
    dataloader = create_data_loaders(args.data_dir, args.batch_size)



    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, dataloader, loss_criterion, optimizer, args.epochs, device, hook)



    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, dataloader['test'], loss_criterion, device, hook)



    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model, path)



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--arch', type=str, default='vgg13', choices=['resnet18', 'vgg13',], help='Load a pre-trained model archictecture (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=5, help='Number epochs for training (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=64, help='Enter number of train batch size (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Enter number of test batch size (default: 32)')
    parser.add_argument('--gpu', type=bool, default=True, help='Enable GPU acceleration for training (default: True)')

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args=parser.parse_args()

    main(args)