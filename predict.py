
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from PIL import Image
import datetime
import argparse
import json
import re

def get_input_args():
    
    # Usage - python predict.py 'flowers/test/4/image_05636.jpg' 'vgg16checkpoint.pth' --gpu --top_k=10
    
    #Output Sample 
    
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('inputImagePath', type = str, default = '', 
                    help = 'path to the image with folder') 
    parser.add_argument('checkpoint', type = str, default = 'vgg16checkpoint.pth', 
                    help = 'checkpoint Path') 
    parser.add_argument('--top_k', type = int, default = 5, 
                    help = 'to return the topk probabilities ') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                    help = 'mapping of categories to real names') 
    parser.add_argument('--gpu', action='store_true', default=False,
                    help = 'Text File with Dog Names') 
  
    return parser.parse_args()

def load_mapping(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(checkpointfile):

    loaded_model = torch.load(checkpointfile)
    # model_name=loaded_model['model_name']
    model_name ='vgg16'
    # model =globals()[model_name](pretrained=True)
    
    model = getattr(models, model_name)(pretrained=True)
    model.classifier = loaded_model['classifier']
#     for param in model.parameters(): param.requires_grad = False
    model.load_state_dict(loaded_model['state_dict'])
#     optimizer.load_state_dict(loaded_model['optimizer'])
    model.class_to_idx = loaded_model['class_to_idx']
    model.classifier = loaded_model['classifier']
#     print(loaded_model.keys())
    model.cuda()
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    loaded_image=Image.open(image)
    loadedImage_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    final_image=loadedImage_transforms(loaded_image)
    return final_image
    
    # TODO: Process a PIL image for use in a PyTorch model

def predict(image_path, model,device,cat_to_name, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    
    image = process_image(image_path).unsqueeze_(0).float()

    #moving the image to cuda
    image=image.to(device)
    #getting the output of the model for the image input
    with torch.no_grad():
        logps = model.forward(image)
    ps = torch.exp(logps)

    #getting the top 5(topk input) probailities-top_prob and classes-top_out from the expoutput 
    top_p, top_name_index = ps.topk(topk, dim=1)
    top_p = top_p.tolist()[0]
    top_name_index = top_name_index.tolist()[0]
    # print(top_p)
    # print(top_name_index)
    index_to_class = {value:key for key, value in model.class_to_idx.items()}
    flower_labels = [cat_to_name[index_to_class[x]] for x in top_name_index]

#     print(flower_labels)
    return top_p, flower_labels


def main():
    args = get_input_args()

    device='cpu'
    if(args.gpu):
        device='cuda'
    print ("device being used is -- "+device)

    cat_to_name=load_mapping(args.category_names)

    model=load_checkpoint(args.checkpoint)

    top_p, flower_labels=predict(args.inputImagePath, model,device,cat_to_name, args.top_k)

    match = re.search(r'/(\d+)/', args.inputImagePath)
    if match:
        folder_index = int(match.group(1))

    input_flower_name=cat_to_name[str(folder_index)]
    print("")
    print("Provided input flower name to the model is---- {} ----".format(input_flower_name))
    print("")
    print("You have requested to find top {} probabilities and below are the top {} probable flowers with probability percentage".format(args.top_k,args.top_k))


    print("")
    for idx, x in enumerate(flower_labels):
        print("Top - "+str(idx+1)+" predicticted flower is -- "+x+" --with probalility - "+ str(round(top_p[idx]*100,2))+" %")
    
if __name__ == '__main__':
    main()