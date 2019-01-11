# 2019/01/11

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



                
class VIS(object):
    """Apply Class Activation Map(CAM) on an image with pre-trained model
    Attr:
        img: PIL image
        model: pre-trained model
        activation_maps: GAP features
        layer_name: name of layer output GAP features
        verbose: verbose mode
    """
    def __init__(self,image,model,verbose=True,figsize=(18,18),columns=5):
        self.img=image
        self.model=model
        #self.model.eval()
        self.verbose=verbose
        self.figsize=figsize
        self.col=columns
        self.output=[image]
        self.fetch() 

    def fetch(self):
        if not isinstance(self.model,str):
            return 
        # fetch model
        self.vprint(" Fetching model <{}> from torchvision ".format(self.model))
        self.model = eval("models."+self.model+"(pretrained=True)")
        self.model.eval()
        # fetch classes
        self.vprint(" Fetching classes from amazon ")
        LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
        self.classes = {int(key):value for (key, value) in requests.get(LABELS_URL).json().items()}
        # process image for any imagenet model
        self.vprint(" Preprocessing image for imagenet model ".format(self.model))
        preprocess = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),\
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.img_var=Variable(preprocess(self.img).unsqueeze(0),requires_grad=True)
        self.img=np.array(self.img)
        
    def vprint(self,s):
        if self.verbose:
            print("####{:#<50}####".format(s))
            
    def forward(self,topk,v=True,):
        self.vprint(" Predicting image ")
        logit = self.model(self.img_var)
        self.logit = self.model(self.img_var)###
        h_x=functional.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        idx=idx.numpy()
        self.idx=idx[0:topk]
        if self.verbose and v:
            for i in range(0,topk):
                print('  {:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))
                
    def ClassActivationMaps(self,topk=4,ratio=0.3,cm=cv2.COLORMAP_JET):
        self.output=[self.img]
        def hook_layer(module,input,output):
            self.activation_maps=output.data.cpu().numpy()
        #self.model._modules.get(target_layer).register_forward_hook(hook_layer) # forward    
        last_conv=[x for x in list(self.model.modules()) if x.__class__.__name__=='Conv2d'][-1]
        assert last_conv.kernel_size==(1,1) 
        last_conv.register_forward_hook(hook_layer)        
        self.forward(topk)
        param=list(self.model.parameters())[-2]
        weight_softmax = np.squeeze(param.data.numpy())
        batch_size, num_channel, h, w=self.activation_maps.shape
        _h,_w,_ = self.img.shape
        for i in self.idx:
            cam=weight_softmax[i].dot(self.activation_maps.reshape(num_channel,h*w))
            cam=cam.reshape(h,w)
            cam=(cam-np.min(cam))*1.0/np.max(cam) # better
            cam=(255*cam).astype(np.uint8)
            heatmap = cv2.applyColorMap(cv2.resize(cam,(_w,_h)), cm)
            result = heatmap*ratio + self.img*(1-ratio)
            result=Image.fromarray(result.astype(np.uint8), 'RGB') # BGR->RGB
            self.output.append(result)
    
    def SaliencyMaps(self,topk=4,cm='hot'):
        self.output=[self.img]
        self.forward(topk)
        h,w,_ = self.img.shape
        for target_class in self.idx:
            self.model.zero_grad()
            model_output=self.model(self.img_var)
            target_output=model_output[0][target_class]
            target_output.backward()
            saliency = self.img_var.grad.data.abs()
            saliency, i = torch.max(saliency,dim=1)  
            saliency = saliency.squeeze().numpy()  
            cm = plt.get_cmap(cm)
            saliency=cm(saliency)[:,:,:3] * 255
            saliency=cv2.resize(saliency,(w, h))
            saliency=saliency.astype(np.uint8)
            self.output.append(saliency)

    def GuidedBackPropagation(self,topk=1,type='norm'):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        update_relus(self.model)
        #layer = get_specific_layer(self.model,layer_name,n)
        #layer.register_backward_hook(hook_function)
        first_layer=[x for x in list(self.model.modules()) if x.__class__.__name__=='Conv2d'][0]
        first_layer.register_backward_hook(hook_function)
        self.output=[self.img]
        self.forward(topk)
        h,w,_ = self.img.shape
        for target_class in self.idx:
            self.model.zero_grad()
            model_output=self.model(self.img_var)    
            target_output=model_output[0][target_class]
            target_output.backward() #gradient=1
            gradient=self.gradients[0].data.numpy()
            if type=='norm':
                gradient-=gradient.min()
                gradient/=gradient.max()
                gradient=np.uint8(255*gradient).transpose(1,2,0)
            elif type=='pos':
                gradient=np.maximum(0, gradient) / gradient.max()
                gradient=np.uint8(gradient.transpose(1,2,0)*255)
            elif type=='neg':
                gradient=np.maximum(0, -gradient) / -gradient.min()
                gradient=np.uint8(gradient.transpose(1,2,0)*255)
            elif type=='gray':
                gradient=convert_to_grayscale(gradient)
                self.a=gradient
                gradient-=gradient.min()
                gradient/=gradient.max()
                gradient=np.uint8(255*gradient).transpose(1,2,0)
                gradient=gradient[:,:,0]
            self.output.append(gradient)
                
    def plot(self):
        h,w,_ = self.img.shape
        self.vprint(" Generating images ")
        plt.figure(figsize=self.figsize)
        columns = self.col
        for i,img in enumerate(self.output):
            plt.subplot(len(self.output) / columns + 1, columns, i + 1)
            plt.axis('off')
            if isinstance(img,Image.Image):
                plt.imshow(img)
            elif len(img.shape)==2:
                plt.imshow(img,cmap='gray')
            else:
                plt.imshow(img)

if __name__=="__main__":
    URL='https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/snake.jpg'
    image = Image.open(io.BytesIO(requests.get(URL).content))
    vis=VIS(image,'resnet152')
    vis.GuidedBackPropagation(type='norm')
    vis.plot()