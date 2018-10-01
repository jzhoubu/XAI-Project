# Convolutional Neural Network Visualizations
My current work is based on [utkuozbulak](https://github.com/utkuozbulak/pytorch-cnn-visualizations)'s work.

## News
- **Update(09/29/2018)**: Add CAM module



## Doing
- Organizing CNN visualizer


## To Do List
- to add README file to CAM
- to seek a small dataset containing clear image to train
- to seek a way to visualize while training

## Gallery

### Class Activation Mapping(CAM)

<table border=0 >
    <tbody>
        <tr>
            <td width="10%" align="center"> Class Activation Mapping </td>
            <td width="18%" > <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_0.jpg"> </td>
            <td width="18%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_1.jpg"> </td>
            <td width="18%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_2.jpg"> </td>
            <td width="18%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_3.jpg"> </td>
            <td width="18%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_4.jpg"> </td>
        </tr>
         <tr>
            <td>  </td>
            <td align="center"> Target class: King Snake (56) </td>
            <td align="center"> Target class: Mastiff (243) </td>
            <td align="center"> Target class: Spider (72)</td>
            <td align="center"> Target class: Spider (72)</td>
            <td align="center"> Target class: Spider (72)</td>
        </tr>
        <tr>
            <td width="10%" align="center"> Colored Vanilla Backpropagation </td>
            <td width="18%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/gradient_visualizations/snake_Vanilla_BP_color.jpg"> </td>
            <td width="18%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/gradient_visualizations/cat_dog_Vanilla_BP_color.jpg"> </td>
            <td width="18%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/gradient_visualizations/spider_Vanilla_BP_color.jpg"> </td>
            <td width="18%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/gradient_visualizations/spider_Vanilla_BP_color.jpg"> </td>
            <td width="18%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/gradient_visualizations/spider_Vanilla_BP_color.jpg"> </td>
        </tr>
    </tbody>
</table>




## Related Work
[utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

[Sample code for the Class Activation Mapping](https://github.com/metalbubble/CAM)

[Data of Places365](http://places2.csail.mit.edu/download.html)

[Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/)

[The Places365-CNNs for Scene Classification](https://github.com/CSAILVision/places365)



