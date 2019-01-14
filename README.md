# Human-eye on Deep Learning

## Background
> Hi, all, thanks for your interest.\
This is my indepedent project at HKUST under supervision of Prof. Huamin Qu during 2018 Fall. 

>In 2018, we've come to the 5th year landmark of deep learning really starting to hit the mainstream. However, for the majority of people outside the academia, deep learning is still remaining as a mystory, a black box algorithm.\
Believing deep learning makes much more sense only when it is trusted for the whole human being community, we build tools to help people visualize and understand what a deep learning model is doing during an image classification task.

## Summary
This repo is designed for **human-eye** to visualize what excatly a CNN network focus on during an image classification task. \
`VIS`  is a tool which helps people visualize features inseide CNN model. This package include three advanced methods: `ClassActivationMaps`, `SaliencyMaps`, and `GuidedBackPropagation`. 

## News
- **Update(09/29/2018)**: Add Class Activation Maps module
- **Update(10/01/2018)**: Add Class Activation Maps demo to Gallery
- **Update(10/02/2018)**: Add Saliency Maps module
- **Update(10/03/2018)**: Add Saliency Maps demo to Gallery
- **Update(10/04/2018)**: Add Guided BackPropagation module
- **Update(10/04/2018)**: Add Guided BackPropagation demo to Gallery
- **Update(10/05/2018)**: Update Research Record
- **Update(11/05/2018)**: Add a practical case for 5002A3 [[Link](https://github.com/sysu-zjw/MSBD-2018Fall/tree/master/5002/A3)]
- **Update(1/12/2019)**: Add Usage module



## Gallery

### Class Activation Mapping  [[Paper](https://arxiv.org/pdf/1512.04150.pdf)]

<table border=0 >
    <tbody>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/bike_0.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/bike_1.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/bike_2.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/bike_3.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/bike_4.jpg"> </td>
        </tr>
         <tr>
            <td align="center" valign="top">  <b>Original image</b> </td>
            <td align="left" valign="top"> <b>Probs</b>: 0.670<br /> <b>Class</b>: {mountain bike, all-terrain bike, off-roader}
            <td align="left" valign="top"> <b>Probs</b>: 0.138<br /> <b>Class</b>: {bicycle-built-for-two, tandem bicycle, tandem}
            <td align="left" valign="top"> <b>Probs</b>: 0.066<br /> <b>Class</b>: {unicycle, monocycle}
            <td align="left" valign="top"> <b>Probs</b>: 0.045<br /> <b>Class</b>: {seashore, coast, seacoast, sea-coast}
        </tr>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/HKUST_0.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/HKUST_1.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/HKUST_2.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/HKUST_3.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/ClassActivationMaps/HKUST_4.jpg"> </td>
        </tr>
         <tr>
            <td align="center" valign="top">  <b>Original image</b> <br />  </td>
            <td align="left" valign="top"> <b>Probs</b>: 0.575<br /> <b>Class</b>: {sundial}
            <td align="left" valign="top"> <b>Probs</b>: 0.098<br /> <b>Class</b>: {palace}
            <td align="left" valign="top"> <b>Probs</b>: 0.046<br /> <b>Class</b>: {bow}
            <td align="left" valign="top"> <b>Probs</b>: 0.045<br /> <b>Class</b>: {stupa, tope}
        </tr>
    </tbody>
</table>

### Saliency Maps  [[Paper](https://arxiv.org/abs/1312.6034)]
<table border=0 >
    <tbody>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/bear_0.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/bear_1.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/bear_2.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/bear_3.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/bear_4.jpg"> </td>
        </tr>
         <tr>
            <td align="center" valign="top">  <b>Original image</b> </td>
            <td align="left" valign="top"> <b>Probs</b>: 0.836<br /> <b>Class</b>: {brown bear, bruin, Ursus arctos}
            <td align="left" valign="top"> <b>Probs</b>: 0.159<br /> <b>Class</b>: {American black bear, black bear, Ursus americanus, Euarctos americanus}
            <td align="left" valign="top"> <b>Probs</b>: 0.003<br /> <b>Class</b>: {sloth bear, Melursus ursinus, Ursus ursinus}
            <td align="left" valign="top"> <b>Probs</b>: 0.001<br /> <b>Class</b>: {wombat}
        </tr>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/mastiff_0.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/mastiff_1.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/mastiff_2.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/mastiff_3.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/SaliencyMaps/mastiff_4.jpg"> </td>
        </tr>
         <tr>
            <td align="center" valign="top">  <b>Original image</b> <br />  </td>
            <td align="left" valign="top"> <b>Probs</b>: 0.938<br /> <b>Class</b>: {bull mastiff}
            <td align="left" valign="top"> <b>Probs</b>: 0.058<br /> <b>Class</b>: {boxer}
            <td align="left" valign="top"> <b>Probs</b>: 0.001<br /> <b>Class</b>: {French bulldog}
            <td align="left" valign="top"> <b>Probs</b>: 0.000<br /> <b>Class</b>: {Brabancon griffon}
        </tr>
    </tbody>
</table>

### Guided BackPropagation  [[Paper](https://arxiv.org/pdf/1412.6806.pdf)]
Some of my codes is implemented based on [utkuozbulak](https://github.com/utkuozbulak/pytorch-cnn-visualizations)'s work. My contributions are listed as below:
- Fix bugs for up-to-date pytorch version
- Solve problems for up-to-date CNN model, eg. ResNet, DenseNet
- Re-organize as a package

<table border=0 >
    <tbody>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/Snake_origin.jpg"> </td>
            <td width="20%" align="center" valign="center">  Guided BackPropagation
            <td width="20%" align="center" valign="center">  Guided BackPropagation with <b>gray scale</b>
            <td width="20%" align="center" valign="center">  Guided BackPropagation with <b>positive gradient</b>
            <td width="20%" align="center" valign="center">  Guided BackPropagation with <b>negitive gradient</b>
        </tr>
        <tr>
            <td width="20%" align="center" valign="center"> <b>DenseNet 161</b>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/DenseNet161_Snake_norm.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/DenseNet161_Snake_gray.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/DenseNet161_Snake_pos.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/DenseNet161_Snake_neg.jpg"> </td>
        </tr>
        <tr>
            <td width="20%" align="center" valign="center"> <b>ResNet 152</b>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/ResNet152_Snake_norm.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/ResNet152_Snake_gray.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/ResNet152_Snake_pos.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/ResNet152_Snake_neg.jpg"> </td>
        </tr>
        <tr>
            <td width="20%" align="center" valign="center"> <b>VGG16_bn</b>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/VGG16_bn_Snake_norm.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/VGG16_bn_Snake_gray.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/VGG16_bn_Snake_pos.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/VGG16_bn_Snake_neg.jpg"> </td>
        </tr>
        <tr>
            <td width="20%" align="center" valign="center"> <b>VGG16</b>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/VGG16_Snake_norm.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/VGG16_Snake_gray.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/VGG16_Snake_pos.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/VGG16_Snake_neg.jpg"> </td>
        </tr>
        <tr>
            <td width="20%" align="center" valign="center"> <b>AlexNet</b>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/alexnet_Snake_norm.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/alexnet_Snake_gray.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/alexnet_Snake_pos.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/GuidedBackPropagation/alexnet_Snake_neg.jpg"> </td>
        </tr>
    </tbody>
</table>

# Usage



```ruby
URL='https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/snake.jpg'
image = Image.open(io.BytesIO(requests.get(URL).content))
vis=VIS(image,'resnet152')   # Class Parameters
vis.GuidedBackPropagation(type='norm') # Method Parameters
vis.plot()
```

```
Class Parameters:
  --image        input image, required PIL image object
  --model       CNN model,  if string, use a pretrained model from torchvision
  --verbose       bool, whether print information during processing
  --figsize       output figure size, default (18,18)
  --columns        output display columns number, default 5

Metod Parameters:
    ClassActivationMaps(topk=4,ratio=0.3,cm=cv2.COLORMAP_JET)
        --topk        number k of topk prediction, default 4
        --ratio        ratio of heatmap against original image, default 0.3
        --cm       type of heatmap, default cv2.COLORMAP_JET
    SaliencyMaps(topk=4,cm='hot')
        --topk        number k of topk prediction, default 4
        --cm       type of colormap, default 'hot'
    GuidedBackPropagation(topk=1,type='norm')
        --topk        number k of topk prediction, default 4
        --type        type of gradient to pass through, default 'norm'.
```

## Reference
[utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

[Sample code for the Class Activation Mapping](https://github.com/metalbubble/CAM)

[Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/)

[DeepTracker: Visualizing the Training Process of Convolutional Neural Networks](http://www.cse.ust.hk/~huamin/tist_2018_dongyu_deeptracker.pdf)


