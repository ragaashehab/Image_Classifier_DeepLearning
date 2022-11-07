from workspace_utils import active_session
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, datasets

from input_args import *
    
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

def process(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                         ])
    # TODO : Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform =train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform =valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform =test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    traindataloader = torch.utils.data.DataLoader(train_datasets, batch_size= 64, shuffle = True)
    validdataloader = torch.utils.data.DataLoader(valid_datasets, batch_size= 64)
    testdataloader = torch.utils.data.DataLoader(test_datasets, batch_size= 64)
    return traindataloader ,  validdataloader , testdataloader

def build_model(arch, hidden_units, learning_rate):
    model = models[arch] #    model = models[model_name]
    #freeze parameters and update classifier
    # Only train the classifier parameters, feature parameters are frozen
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(hidden_units, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    return model

# Main program function defined below
def main():    
    # TODO 1: Define get_train_input_args function within the file get_input_args.py
    in_arg = get_train_input_args()
    # Function that checks command line arguments using in_arg  
    check_command_line_arguments(in_arg)
    
    # TODO 2: Load and define datasets directories, transforms, ImageFolder, and dataloaders
    train_loader , valid_loader , test_loader = process(in_arg.dir)
    # test the data loader
    images, labels = next(iter(train_loader))
    #print(images.shape, labels.shape, images.type)
        
    # TODO 3-1: Build your network 
    model = build_model(in_arg.arch, in_arg.hidden_units, in_arg.learning_rate)
    #freeze parameters and update classifier
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
   
    # TODO 3-2: train your network  
    # use GPU if available
    device = torch.device("cuda" if (torch.cuda.is_available() and in_arg.gpu) else "cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device " , device)
    model.to(device);
    #print(model.classifier.state_dict)   
    steps = 0
    running_loss = 0
    print_every = 5
    with active_session():
        # do long-running work here
        for epoch in range(in_arg.epochs):
            for inputs, labels in train_loader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model(inputs)  # forward path
                loss = criterion(logps, labels)  # calculate loss 
                loss.backward()         # backpropagation , auto grade: propagate back loss to obtain weights (w_dash)
                optimizer.step()         # calculate new weights according to learning rate(w_new)

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()    # stop drop out , 
                    with torch.no_grad():   # stop auto grade to speed up
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model(inputs)
                            batch_loss = criterion(logps, labels)
                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)  # turn model output into probabilities
                            top_p, top_class = ps.topk(1, dim=1)   # find first top prob and it's equivelant top class
                            equals = top_class == labels.view(*top_class.shape)  #if top_class == labels ,let equals=1
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                        print(f"Epoch {epoch+1}/{in_arg.epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"valid loss: {valid_loss/len(valid_loader):.3f}.. "
                              f"valid accuracy: {accuracy/len(valid_loader):.3f}")
                    running_loss = 0
                    model.train()
                    
    
    # TODO 3-3: Do validation on the test set
    with active_session():
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"test loss: {test_loss/len(test_loader):.3f}.. "
                  f"test accuracy: {accuracy/len(test_loader):.3f}")

    # TODO 4: Save checkpoint                
    checkpoint = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': in_arg.epochs,
                  'learning_rate': in_arg.learning_rate,
                  'arch': in_arg.arch,                 
                  'hidden_layers': in_arg.hidden_units,
                  'GPU': in_arg.gpu,
                  'class_to_idx': class_to_idx(in_arg.cat_names)
                  }

    torch.save(checkpoint, in_arg.checkpoint)

# Call to main function to run the program
if __name__ == "__main__":
    main()