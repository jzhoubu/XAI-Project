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

%matplotlib inline

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
        self.verbose=True
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
                
    def ClassActivationMaps(self,target_layer,topk=4,ratio=0.3,cm=cv2.COLORMAP_JET):
        self.output=[self.img]
        def hook_layer(module,input,output):
            self.activation_maps=output.data.cpu().numpy()
        self.model._modules.get(target_layer).register_forward_hook(hook_layer) # forward
        self.forward(topk)
        param=list(self.model.parameters())[-2]
        weight_softmax = np.squeeze(param.data.numpy())
        batch_size, num_channel, h, w=self.activation_maps.shape
        _h,_w,_ = self.img.shape
        for i in self.idx:
            cam=weight_softmax[i].dot(self.activation_maps.reshape(num_channel,h*w))
            cam=cam.reshape(h,w)
            cam=(cam-np.min(cam))*1.0/np.max(cam) # better
            #cam=cam-np.min(cam)
            #cam/=np.max(cam)
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
            if len(img.shape)==2:
                plt.imshow(img,cmap='gray')
            else:
                plt.imshow(img)