# CarND Behavioral Cloning
## by Guntis Valters

This readme has the following structure:
1. Overall approach<br/>
2. Challenges<br/>
2.1. Input Data gathering<br/>
2.2. Number of various parameters<br/>
2.3. Track configuration<br/>
3. Model network architechture<br/>
3.1. Why initial 3x1x1 convolution layer<br/>
3.2. Why VGG type network<br/>
3.3. Why stacking intermediate layers<br/>
3.4. Why Adam optimizer<br/>
4.5. Number or epochs<br/>
4. Data preparation<br/>
4.1. Data cleanup<br/>
4.2. Camera selecion<br/>
4.3. Region selection and downsampling<br/>
4.4. Normalization<br/>
5. Reflection<br/>

## Overall approach
This challenge was something I was looking forward to experiment with, especially because there was simulator provided that allowed easily to record the images and afterwards run the code on the same environment.<br/>
Finishing the traffic sign classifier I was impressed how well the convolutional networks worked and how simple it is to implement them if there is proper input data to work on.<br/>
Initially I followed approach that created small model, gathered (10Hz) input with analogous joystick and tried to work on the central camera. By training with my collected data and improving the model, I tried several approaches, however one of the best results was achieved with VGG type network, where there are multiple convolution layers with small kernel (3x3 in my case) and following maxpool layer that reduces the dimensions. With 2x2 convolution layers and 3 fully connected layers I got better results, however still was not able to simulate reliably and the best results were to drive over the bridge, but following the dirt path on the right.<br/>
I tested then the same model on the Udacity provided dataset, which provided much smoother (not so oscilalting) results. By comparing the dataset provided by Udacity and my own recorded one, I saw that I have recorded many locations with 0 steering, which did not provide good enough information for the model to train on.<br/>
**Decision at this point was to go with Udacity dataset** in order to focus on model and processing pipeline and to eliminate one very significant variable which is input data.<br/>
After that focus was to get the car all around the track by testing various models and adjusting them.<br/>

## Challenges
Number of challenges was faced during the project:
### Input Data gathering
Input data proved to be more significant then initially expected. My expectation was that it woul dbe enough to drive several times around the circuit and keeping on the rand and that would be enough to train the model. However, later when looking at how the model behaves, I understood that there was a lot of data that was incorrect and led to training wrong behavior. For example, parts of the curved sections was driven straight with 0 steering angle or even adjustments made to wrong direction. If model took such data for training then it produced worse result in the simulator. Same applied to adjusting on the straight track when te car is in the middle, it led simulated car to drive away from centerline. This was resolved by taking better input data (Udacity dataset) and running additional cleanup (described in chapter "Data cleanup")


### Number of various parameters
Model consisted on many parameters that each affected the model behavior when simulating (number or epochs, model architechture, layer sizes, etc). Challenge with that was, that there was no good measure how to evaluate model accuracy. In comparison, during traffic sign classifier, if the loss was getting smaller, the better was the model.<br/>
However, in this behavior simulation, there was no good measure, because even if validation set was performing well, it did not guarantee predictable behavior in all track locations, it ensured that for most of the trach it behaves better.

### Track configuration
When driving the track by myself, it looked as very simple track and no tricky places, moreover, all thack could be driven with full throttle without any issues. However, when started to simulate, then several tricky places quickly was discovered:<br/>
**Bridge:**<br/>
![Bridge](https://github.com/Valtgun/CarND_Behavioral_Cloning/blob/master/readme_img/center_2016_12_01_13_44_43_549.jpg)<br/>
Due to the different color scheme (black sides) from the rest of the track, it was initially when simpliest models did not perform well, however introducing more layers and dropouts helped to overcome this issue quickly.<br/>
**Left turn after bridge with dirt area on the right**<br/>
![Left turn after bridge with dirt area on the right](https://github.com/Valtgun/CarND_Behavioral_Cloning/blob/master/readme_img/center_2016_12_01_13_31_13_890.jpg)<br/>
This turned out to be the biggest challenge all over the track, because model wanted to go right on the dirt and not make the sharp left turn. With additional data cleanup and changing the camera data used, it was possible to overcome this location.<br/>
**Right (2nd) turn after the bridge**<br/>
![Right (2nd) turn after the bridge](https://github.com/Valtgun/CarND_Behavioral_Cloning/blob/master/readme_img/center_2016_12_01_13_45_06_414.jpg)<br/>
This became sign for the model to start overfitting. If model was run over more epochs, then it 'wanted' to drive more smoothly and forward and did not do so well on the sharp right curve. This was solved with more dropout layers and data cleanup.

## Model network architechture
![Model architechture](https://github.com/Valtgun/CarND_Behavioral_Cloning/blob/master/readme_img/model.png)
Final Architechture was a result of iterative approach. Initially several models was tested and one of the best results was with the above model. Later 3rd convolution layer was added as well as intermediate connections to first fully connected layer.

### Why initial 3x1x1 convolution layer
First layer of the network was 3x1x1 convolution that I used also on the traiffic sign classifier, in order for the model to be able to adjust parameters for the colorspace. As the conversion between various colorspaces, e.g. RGB to HSL is just multiplication of each channel to get the new color channel then this layer is added for the model to train such behavior if necessary.

### Why VGG type network
I like the VGG type network when there are convolution layers that has small kernel and does not affect dimensions with a mix of maxpool layers to reduce dimensions, because the model is quite straight forward in terms of input and output shapes. Each high level in this network reduces dimensions by two and has inside several convilution layers.<br/> For me this model worked very well on trafic sign classifier and also turned out to have good results on this assignment.<br/>
One consideration is that as the input image was not of the dimensions of power of 2, it had reduction from dimansion of 5 to dimension of 2 in last maxpool layer. The loss of data was somewhat compensated with stacking also intermediate layers.

### Why stacking intermediate layers
Results from the intermediate maxpool layers also was passed to final fully connected layer. This was done in order not to loose valuable features if there are such in the middle of the network. This gave improved results, especially in the left turn after the bridge. After maxpool layer data is passed to next convolution layer and as well it is flattened and passed to first fully connected layer.

### Why Adam optimizer
Also in this assignment Adam optimizer was chosen, because it has less hyperparameters to tune (and make errors). I am still at the stage, when I am learning models and generic approach and even without such hyperparameters there is a lot of other parameters to tune. I think chould of optimizer owuld be very important when planning to get improvements from 95% to 98+%, however I am still in a place, where I need to get 80%-90% reliably.

### Number or epochs
In the final solution, number of epochs is 5. Larger values that that, started to overfit the model and in simulator it meant that it preferred to drive straight, therefore starting to miss sharp turns. Less that that and simulator mdel oscellated more from side to side over the road.

## Data preparation
As well as in the traffic sign classifier, this turned out to be most significant factor for sucessful model.
### Data cleanup
From the Udacity dataset there was some additional cleanup made in the regions where model wanted to behave not according to intention. I checked the images whre there is right turn with the red and white borders (to improve results on the 2nd corner after the bridge), fo rthose images, if the steering angle was 0, then they were removed completelly from the input dataset.<br/>
Also similar was done to the images for 1st left turn after the bridge, if the angle was 0 then they were removed. This allowed to navigate those corners more sharply and drive the track.

### Camera selecion
Initially the model was trained on center camera, however it was oscillating quite a lot in the middle of the track and there was very poor recovery. I decided to add also left and right image, with the angle adjustment of 0.08.
This parameter was tuned by trial and error. Initially chosen larger parameter, and then lowered it so that critical corners are still navigated succesfully. The larger the parameter the more oscillation also on the straight track, but the recovery is improved.<br/>
Also tested if only the left and right camera is used without the central camera. This turned out to have the best results. The oscillation was lower that training with all 3 cameras and much better recovery. My hypothesis is that this is because there is a lot of camera data with 0 steering angle, but slightly different car alignments to the track axis. This slightly varying data confuses model training and provides more erratic weights. If there is only left and right camera data with angle adjustment of +/-0.08, then there is almost no data with 0 steering angle.<br/> Possibly this could be also done with altering initial dataset and removing 0 steering data, but that would require aditional data collection.

### Region selection and downsampling
Vertical region from above car and below around horizon was chosen with full width.<br/>
Also the each second pixel was taken in order to reduce image dimensions.

### Normalization
Input RGB images of uint values of 0-255 was normalized to -1.0 to 1.0 floating values. In the traffic sign classifier I also observed that if the values are normalized then training performs better (gets accuracy quicker) as the weights are more uniform.

## Reflection
This was a very interesting project. I have investigate multiple architechtures and still remained with the VGG approach that I used also in the traffic sign classification with the adjustment that also intermediate layers are connected to first fully connected layer. There was more challenged that I expected initailly, but that allowed me to better understand how the convolutional networks operate.<br/>
I value this a lot as a learning experience and that I was able to reach result of driving the lap sucessfully. This model definitelly could be improved and especially with the addtional data that can be collected and properly annotated.<br/>
In my opinion this also shows why the data is so important and the more date is being gathered by some party, the mode possibilities they have to create better model.
