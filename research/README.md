# Research on Visualization on CNNs

## News

### 09/29/2018
Read [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf) and reproduced CAM demo
> My understanding: With global average pooling(GAP), it will be filter-level but no more pixel-level to contribute to final prediction in a CNN. In this case, the weights of fully connected layer can be considered as a  non-linear combination of the feature-maps that right before GAP layer. By using this weights on feature map, they surprisedly found the mask can do weakly-supervised object localization and visualization on the input image.

> 一个用了GAP的CNN网络结构在进行预测的时候，filter中的每一个像素点对class的score的贡献程度是一样的（这是因为GAP在softmax前将filter变成了一个像素点），这意味着对softmax layer的权重是作用在filter level的，而不是pixel level。这篇文章将softmax layer的weight直接作用在最后的filter上，可以理解为多个filter的线性组合得到一个mask，套在原图上有可视化和焦点定位的作用。\
> eg. 以`densenet161`为例，其最后一个卷积层的输出的维度为(2208, 7, 7)。那么，在进行训练和预测的时候，(2208, 7, 7)通过GAP后维度变成(2208,1,1)再进入softmax层(2208,1000)输出分数。而在生成CAM的时候我们只关注某一个分类的权重W，维度为(2208,1)，这个W可以理解为filter对这个class预测的贡献的大小。用这个权重对2208个(7,7)的filter进行线性组合即得到CAM。

### 09/30/2018
Read [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) 

> 用输出的score对input image（记为X）求导，导数记为W，维数与X相同。W取绝对值，取channel的最大值进行降维，可以作为一个mask套在原图上。这个操作可以取CNN网络的任意layer的output进行反向求导。paper的interpret是，将X展开成向量，将W作为泰勒展开式的一阶导。

### 10/02/2018
Reproduced Saliency Maps module





