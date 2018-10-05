# Human-EYE on Deep Learning

## Background
Hi, all, thanks for your guys interest. This is my indepedent project at HKUST under supervision of Prof. Huamin Qu during 2018 Fall. \
In 2018, we've come to the 5th year landmark of deep learning really starting to hit the mainstream. However, for the majority of people outside the academia, deep learning is still remaining as a mystory, or a black box algorithm. Believing deep learning, a high level algorithm associated with math and computer science, makes much more sense when it is trusted for the whole human community, we make tools to help people visualize and better understand what deep learning is doing about during an image classification task.

## Summary
This repo is designed for human-eye to visualize what excatly a CNN network focus on during an image classification task. \
Further, I would interpret how these visualizer works with CNN architectures, and struggle on better improvement.

- Doing 
    - Organizing CNN visualizer
        + CAM(Done)
        + Saliency Maps(Done)
        + Guided BackPropagation(Done)
- To Do
    - to add `Usage` module in README
    - to discover why guided BP work better (related to Gabor-like filters, `neuron would be activated the most if the input looks like the weight matrix`)
    - to discover ways for guided BP in densenet
    - to seek a way to visualize while training
    - to seek a small dataset containing clear image to train (Removed)


## News
- **Update(09/29/2018)**: Add Class Activation Maps module
- **Update(10/01/2018)**: Add Class Activation Maps demo to Gallery
- **Update(10/02/2018)**: Add Saliency Maps module
- **Update(10/03/2018)**: Add Saliency Maps demo to Gallery
- **Update(10/04/2018)**: Add Guided BackPropagation module
- **Update(10/04/2018)**: Add Guided BackPropagation demo to Gallery
- **Update(10/05/2018)**: Update Research Map


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

### Guided BackPropagation  [[Paper](https://arxiv.org/abs/1312.6034)]
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



## Related Work
[utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

[Sample code for the Class Activation Mapping](https://github.com/metalbubble/CAM)

[Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/)

[DeepTracker: Visualizing the Training Process of Convolutional Neural Networks](http://www.cse.ust.hk/~huamin/tist_2018_dongyu_deeptracker.pdf)


