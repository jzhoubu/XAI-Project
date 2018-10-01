# Convolutional Neural Network Visualizations
My current work is based on [utkuozbulak](https://github.com/utkuozbulak/pytorch-cnn-visualizations)'s work.

## News
- **Update(09/29/2018)**: Add CAM module



## Summary
- Doing 
    - Organizing CNN visualizer
        + CAM(done)
        + Saliency Maps
- To Do
    -  to add README file to CAM
    - to seek a small dataset containing clear image to train
    - to seek a way to visualize while training

## Gallery

### Class Activation Mapping(CAM)

<table border=0 >
    <tbody>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_0.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_1.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_2.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_3.jpg"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/XAI-Project/blob/master/images/CAM/bike_4.jpg"> </td>
        </tr>
         <tr>
            <td align="center" valign="top">  <b>Original image</b> </td>
            <td align="left" valign="top"> <b>Probs</b>: 0.670<br /> <b>Class</b>: {mountain bike, all-terrain bike, off-roader}
            <td align="left" valign="top"> <b>Probs</b>: 0.138<br /> <b>Class</b>: {bicycle-built-for-two, tandem bicycle, tandem}
            <td align="left" valign="top"> <b>Probs</b>: 0.066<br /> <b>Class</b>: {unicycle, monocycle}
            <td align="left" valign="top"> <b>Probs</b>: 0.045<br /> <b>Class</b>: {seashore, coast, seacoast, sea-coast}
    </tbody>
</table>




## Related Work
[utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

[Sample code for the Class Activation Mapping](https://github.com/metalbubble/CAM)

[Data of Places365](http://places2.csail.mit.edu/download.html)

[Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/)

[The Places365-CNNs for Scene Classification](https://github.com/CSAILVision/places365)



