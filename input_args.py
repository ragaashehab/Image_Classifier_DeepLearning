#PURPOSE: Create a function that 
#          retrieves the command line inputs 
#          check input args
#           turn class index into category
#          turn category index into name
##
# Imports python modules
import argparse
from os import listdir
import json


# TODO 1: Define get_train_input_args function get train.py input

def get_train_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the train.py from a terminal window. 
    """
     # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Retrieves command line arguments provided by the user to train.py.')
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--dir', default= 'flowers', help='data folder')
    parser.add_argument('--arch', default= 'vgg', help='CNN Model Architecture')
    parser.add_argument('--flcat', default= 'cat_to_name.json' , help='flowers categories to index mapping file')
    parser.add_argument('--checkpoint', default= 'checkpoint.pth', help='checkpoint save_directory')
    parser.add_argument('--learning_rate', default= 0.001 , help='learning_rate')
    parser.add_argument('--hidden_units', default= 4096 , help='hidden_units')
    parser.add_argument('--epochs', default= 1 , help='epochs')
    parser.add_argument('--gpu', default= True , help='GPU')
 
    args = parser.parse_args()
    return args  

def get_predict_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the predict.py program from a terminal window. 
    """
     # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Retrieves command line arguments provided by the user.')
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--image',default= 'flowers/test/101/image_07949.jpg', help='image to test folder')
    parser.add_argument('--checkpoint',default= 'checkpoint_1.pth', help='checkpoint used in test')
    parser.add_argument('--arch',default= 'vgg', help='model name')
    parser.add_argument('--top_k', default= 3 , help='top K most likely classes')
    parser.add_argument('--cat_names', default= 'cat_to_name.json' , help='flowers categories to index mapping file')
    parser.add_argument('--gpu', default= True , help='GPU')
    parser.add_argument('--dir', default= 'flowers', help='data folder')

 
    args = parser.parse_args()
    return args  
    
    
def check_command_line_arguments(in_arg):
    """
    For Lab: check_command_line_arguments: Prints each of the command line arguments passed in as parameter in_arg, 
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print (in_arg)

        
def cat_to_name(cat_to_name):
    # category label to category name mapping
    with open(cat_to_name, 'r') as f:
        cat_to_name_dic = json.load(f)
    return cat_to_name_dic   #imagenet_classes_dict

def test_class_to_idex(data_dir): #image_dir
    # Retrieve the filenames from folder pet_images/
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    class_to_idx_lst = sorted(listdir(test_dir)) #data_dir + "/train")  # "flowers/" flwscateg_list = [1 to 102]
    test_class_to_idx_dic = {}
    for i in range(len(class_to_idx_lst)):
        test_class_to_idx_dic[i] = class_to_idx_lst[i]
    print(len(class_to_idx_lst), test_class_to_idx_dic)
    return test_class_to_idx_dic

#class_to_idx = class_to_idx('flowers')
