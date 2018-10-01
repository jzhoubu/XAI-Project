# Research on Visualization on CNNs

## News

### 09/29/2018
Read [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf) and reproduced CAM demo
> My understanding: With global average pooling(GAP), it will be filter-level but no more pixel-level to contribute to final prediction in a CNN. In this case, the weights of fully connected layer can be considered as a  non-linear combination of the feature-maps that right before GAP layer. By using this weights on feature map, they surprisedly found the mask can do weakly-supervised object localization and visualization on the input image.

### 09/30/2018



### Learning Deep Features for Discriminative Localization([Link](https://arxiv.org/pdf/1512.04150.pdf))
- **Code Demo**: https://github.com/metalbubble/CAM
- **Contributions** of this work:
    -  Weakly-supervised object localization
    -   Visualizing CNNs    
- **Related Work** worth mentioning: 
    - [Visualizing and Understanding Convolutional Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)  (to-do-list) uses deconvolutional network to visualize patterns each unit


