# **Behavioral Cloning Project** 
---

[//]: # (Image References)

[img1]: ./writeup_images/dataset_combined.png "Dataset"
[img2]: ./writeup_images/fitting_epochs.png "RSE"
[img3]: ./writeup_images/cropping.png "Crop"


## Instructions:

### Training the model:
 
 - Provide the relative paths to the data folders in `parameters.py``(DATASET_FOLDERS), and execute:


``` 
python model.py 
```

### Driving using the model:
 - Open the simulator in ''Autonomous Mode" and execute:

``` 
python drive.py model.h5
```


## Model Architecture and Training Strategy

### Dataset

First the dataset was explored, it's very important to note that the steering angle recorded alongside the images is normalized, a value of 1.0 corresponds to a 25º angle while in training mode. Also, because the network is trained with these values it means that the maximum angle change between frames is restricted to +/- 1º this has two major consequences:
 
 - Helps reducing steering jerks in straight segments. 
 - It restricts the vehicle in recovery scenarios, since it can only adjust 1º per frame, nevertheless it's sufficient to drive through the sharpest curves on both tracks. 
 
Additional data were collected from both tracks, using the mouse instead of the keyboard since it allows for a better continuous control. The control provided by the keyboard has a more discrete nature which influences negatively the fitting process. The following scenarios for data collection were used:
	
 - `track1`: 2 laps centered, 1 lap reverse, 1 lap on left recovery, 1 lap on right recovery;
 - `track2`: 2 laps centered, 1 lap reverse.

Histograms of the dataset are presented below:

![alt text][img1]

Remarks:

 - The majority of samples are around 0º and +/- 5º, mostly due to the long curved segments in  `track1
 - The data for sharp curves (+/- 25º) is much more scarce, by a factor of 10^3
 - A correction offset of `0.2` (+/- 5º) was introduced for left and right images which is visible  on the histogram. 
	

### Data augmentation

A simple horizontal mirror is applied in every image alongside with inverting the steering angle direction. The input dataset used has `58k` samples, with data augmentation this number is doubled. 

### Preprocessing

Just a simple cropping on top and bottom was used to grab the relevant portion of the track.

![alt text][img3]

 - Sample image with `steering = -0.21`

### Model

The NVIDIA model for autonomous driving vehicles, described [here](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), was used. The architecture summary follows:

```
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```
The model has close to 1M parameters which seem too much for driving along two simple tracks, a way to dramatically decrease the number of parameters is by decreasing the input size. Resizing the image to a smaller size or using a single color layer (saturation) are two good options, however it was not performed in this project. 

Mean Squared Error was used has the loss function, along with the Adam optimizer. 

To find an optimal value for the number of epochs used in the training process, the MSE of the training and validation sets were plotted. Results bellow:

![alt text][img2]

It's visible that 3-4 epochs are the best values since they best prevent both underfitting and overfitting. 


## Results


Videos of the laps using the provided model available here: [Track1](./video_track1.mp4) and [Track2](./video_track2.mp4).

The vehicle is able to successfully complete the two tracks in safety. However, especially in `track2` by displacing manually the vehicle before sharp curves and thus changing the attack angle, the vehicle understeers and goes out of track. This could be improved by providing a more uniform dataset, in this case by providing more images where the steering angle is higher than 15º (mod).


