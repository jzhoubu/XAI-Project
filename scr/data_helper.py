import torch
from torchvision import models, transforms
from torch.nn import functional
import cv2
import io
import requests
from PIL import Image
from torch.autograd import Variable
import numpy as np
import pdb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from torch.nn import ReLU


def relu_hook_function(module, grad_in, grad_out):
    """If there is a negative gradient, changes it to zero
    """
    if isinstance(module, ReLU):
        return (torch.clamp(grad_in[0], min=0.0),)
    
def update_relus(model):
    for module in model.modules():
        if isinstance(module, ReLU):
            module.register_backward_hook(relu_hook_function)

def get_first_conv_layer(model):
    return [x for x in list(model.modules()) if x.__class__.__name__=='Conv2d'][0]

def get_specific_layer(model,layer_name='Conv2d',n=0):
    return [x for x in list(model.modules()) if x.__class__.__name__==layer_name][n]

def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im