import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import datetime
import argparse

def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, default = 'flowers', 
                    help = 'path to the folder of flower images') 
    parser.add_argument('--save_dir', type = str, default = 'vgg16', 
                    help = 'path to the folder to save the pth file') 
    parser.add_argument('--arch', type = str, default = 'vgg16', choices =['vgg16','vgg19'],
                    help = 'CNN Model Architecture') 
    parser.add_argument('--learning_rate', type =float , default = '0.002', 
                    help = 'learning rate for the learing') 
    parser.add_argument('--hidden_units', type = int, default = 408, 
                    help = 'how many hidden units in the arcitecture') 
    parser.add_argument('--epochs', type = int, default = 3, 
                    help = 'epochs') 
    parser.add_argument('--gpu', action='store_true', default=False,
                    help = 'Text File with Dog Names') 
  
    return parser.parse_args()

def load_tansforms(data_dir):
#     data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets

    training_transforms =  transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406] ,
                                                              [0.229, 0.224, 0.225])])

    testing_transforms= transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406] ,
                                                              [0.229, 0.224, 0.225])])

    validatoion_trasforms= transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406] ,
                                                              [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    #image_datasets = 

    training_data=datasets.ImageFolder(train_dir, transform=training_transforms)
    testing_data=datasets.ImageFolder(test_dir, transform=testing_transforms)
    validation_data=datasets.ImageFolder(valid_dir, transform=validatoion_trasforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = 

    trainloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testing_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    
    return trainloader,testloader,validloader,training_data

def create_model(arch,hidden_units,learning_rate,device):
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='vgg19':
        model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    input_count=model.classifier[0].in_features
    fc2output= int( hidden_units/2)
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_count, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, fc2output)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(fc2output, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion= nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    # print(model)
    return model,optimizer,criterion
         
def train_model(model,epochs,trainloader,validloader,device,optimizer,criterion):
#     epochs = 3
    
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        print("in epoch,"+ str(epoch))
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"time now {datetime.datetime.now()}.."
                      f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}") 
                running_loss = 0
                model.train()
    return model
                
def test_model(model,testloader,device,criterion):
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    
def save_model(model,save_path,training_data,optimizer,epochs,arch):
    model.class_to_idx = training_data.class_to_idx

    attributes= {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                 'class_to_idx': model.class_to_idx,
                'epohcs':epochs,
                'model_name':arch,
                'classifier': model.classifier
                }
#     torch.save(attributes, 'vgg16.pth')
    torch.save(attributes, save_path)

def main():
    args = get_input_args()

    trainloader,testloader,validloader,training_data = load_tansforms(args.data_dir)
    device='cpu'
    if(args.gpu):
        device='cuda'
    print ("device being used is"+device)
    print("1")
    model,optimizer,criterion=create_model(arch=args.arch,hidden_units=args.hidden_units,learning_rate=args.learning_rate,device=device)
    print("2")
    model=train_model(model,args.epochs,trainloader,validloader,device,optimizer,criterion)

    test_model(model,testloader,device,criterion)
    print("3")
    save_model(model,args.save_dir+'checkpoint.pth',training_data,optimizer,args.epochs,args.arch)
    print("4")
    
if __name__ == '__main__':
    main()