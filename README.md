# Convolutional Neural Network Visualizations
My current work is based on [utkuozbulak](https://github.com/utkuozbulak/pytorch-cnn-visualizations)'s work.

## News
- **Update(09/29/2018)**: Add CAM module
- **Update(10/01/2018)**: Add CAM example into Gallery


## Summary
- Doing 
    - Organizing CNN visualizer
        + CAM(Done)
        + Saliency Maps(Done)
- To Do
    -  to add README file to CAM
    - to seek a small dataset containing clear image to train
    - to seek a way to visualize while training

## Gallery

### Class Activation Mapping(CAM)

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

### Saliency Maps
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

## Related Work
[utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

[Sample code for the Class Activation Mapping](https://github.com/metalbubble/CAM)

[Data of Places365](http://places2.csail.mit.edu/download.html)

[Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/)

[The Places365-CNNs for Scene Classification](https://github.com/CSAILVision/places365)



