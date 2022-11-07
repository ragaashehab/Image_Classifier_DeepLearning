from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__ 
import torch
from torch import nn
from torch import optim

from input_args import * 

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

def load_checkpoint(checkpoint_filepath):
    checkpoint = torch.load(checkpoint_filepath)
    arch = checkpoint['arch']
    epochs =checkpoint['epoch']
    lr = checkpoint['learning_rate']
    h_layer= checkpoint['hidden_layers']
    gpu= checkpoint['GPU']
    #class_to_idx = checkpoint['class_to_idx']
    
    model = models[arch]
    
    classifier = nn.Sequential(nn.Linear(25088, h_layer),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(h_layer, h_layer),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(h_layer, 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict((checkpoint['optimizer']))
    return model

    
def main():    
#def predict():
#def predict(img_path, checkpoint_path):  def predict(img_path, model_name):
    in_arg = get_predict_input_args()
    # Function that checks command line arguments using in_arg  
    check_command_line_arguments(in_arg)
    #load_checkpoint(in_arg.checkpoint)

    # load the image
    img_pil = Image.open(in_arg.image)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')
    #print("pytorch_ver" , pytorch_ver)
    
    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the 
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set 
    # requires_grad_ to False on our tensor 
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
    
    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True) 

    # apply model to input
    #model = models[in_arg.arch] #    model = models[model_name]
    model= load_checkpoint(in_arg.checkpoint)

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()
    
    # apply data to model - adjusted based upon version to account for 
    # operating on a Tensor for version 0.4 & higher.
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor)
        
    # pytorch versions less than 0.4
    else:
        # apply data to model
        output = model(data)
      
    ps = torch.exp(output)
    top_p , top_class = ps.topk(in_arg.top_k, dim=1)
    pred_idx = output.data.numpy().argmax()
    
    test_class_to_idx= test_class_to_idex(in_arg.dir)
    pred_class= test_class_to_idx[pred_idx]
    print("pred_idx , pred_class ", pred_idx , pred_class)
    pred_class =  str(pred_class)    
    # obtain ImageNet labels, category label to category name mapping
    imagenet_classes_dict= cat_to_name(in_arg.cat_names)  # image category to image name dictionary
    
    print ("top_p", top_p , "top_class ", top_class , "pred_idx " , pred_idx , " predict output ", imagenet_classes_dict[str(pred_class)])
    #pred_class= class_to_idx_lst[pred_idx]
    #pred_class =  str(pred_class)
    #print ("class_to_idx_lst", class_to_idx_lst , "imagenet_classes_dict ", imagenet_classes_dict) 
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
    