#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/center_full2.png "Center Image"
[image9]: ./examples/left_full2.png "Left Image"
[image10]: ./examples/right_full2.png "Right Image"
[image11]: ./examples/hist3.png "Histogram"
[image12]: ./examples/center_small.png "Center small"
[image13]: ./examples/left_small.png "Left small"
[image14]: ./examples/right_small.png "Right small"
[image15]: ./examples/center_luminance.png "Center lum"
[image16]: ./examples/left_luminance.png "Left lum"
[image17]: ./examples/right_luminance.png "Right lum"
[image18]: ./examples/center_equlized.png "Center equi"
[image19]: ./examples/left_equlized.png "Left equi"
[image20]: ./examples/right_equlized.png "Right equi"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I investigated the pre processing schemes like augmentation, cropping, luminance extraction and brightness correction outside the model. Finally I used LeNet architecture for image to steering regression.

The network is defined in model.py from lines 132 till 158. My final model consists of two convolutional layers, followed by a flatten, followed by two fully connected layers and finally instead of 10 outputs a single output for steering angle. A detailed description of the network and changes done for the problem in hand are explained in the next section.



####2. Attempts to reduce overfitting in the model

Data has been split into training and validation sets at line 42. 20% of the data is used for validation. Since LeNet is a small model and is very unlikely to be overfit in case of enough data provided. Moreover enough preprocessing on images made sure that network didn't need to learn everything from scrtach which helped against underfitting.

The validation set accuracy and loss were equally good as the training set which is a proof that the network is neither overfitting nor underfitting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 157).

####4. Appropriate training data

After landing multiple times in the lake at the first track I decided to use Udacity data. Data was further augmented and preprocessed before used for training the network.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started by visual investigation of the training images and corresponding steering angles. Three cameras capture the scene synchronously however they have a certain offset because of the position on the car. Follwing are the images of the same scene from three different cameras:-

![alt text][image8]
![alt text][image9]
![alt text][image10]

I added an offset angle (model.py line 91) to the steering angle for left cam images and subtracted the same offset (model.py line 98) for the angles from the right camera images. First of all it helped as it added non zero values to the target. Moreover the total unique steering angles were raised from 124 to 372. However the distribution of the angles was skewed which would mean that car was likely to turn more towards one side. To compensate this skewness all the dataset was horizontally flipped and angles were multiplied to -1. It increased the training data size twice as well as the distribution was even on both sides of 0°. Number of unique angles raised upto 741. Follwing is the histogram of steering angles of the final training data.

![alt text][image11]

Regression at this stage can be seen as a classification problem with 741 unique labels. Steering angle of the car depends on the road curvature so it can follow the road without touching the road boundary. The most relevant information is hence the road curvature. In our image set it is confined in 32 pixels starting horizontally from pixel 80. Moreover vertical pixels are also downsampled from 320 to 32 pixels.

Follwing are the images  of the same scene as shown above from left to right cam displayed in the same order cropped to 32x32x3 pixels.

![alt text][image13]   ![alt text][image12]   ![alt text][image14]

 As mentioned above the most important information in the road images is the curvature which is shown by lines so the color information is not so relevant. To save the processing time and reducing the camera cost the images are coverted to Luminance and chrominance color space (model.py line 57). From this step onwards only luminance part of the images is used for further processing. Following is the luminance of the same images from all three cameras.

 ![alt text][image16]   ![alt text][image15]   ![alt text][image17]

It was observed that the brightness of the street differes at different locations because of shadows and direct sun. To compensate for this unwanted brightness effect histogram equlization was applied to the images. Following is the same set with histogram equlization.

![alt text][image19]   ![alt text][image18]   ![alt text][image20]

The images are then normalized to 0 mean and maximum deviation of 0.5 to both positive and negative.



####2. Final Model Architecture

The final model architecture (model.py lines 132-158) consisted of a LeNet netowork with the following layers and layer sizes:-

| Layer         		|     Description	        					|Parameters|
|:---------------------:|:---------------------:|:------------------------:|
| Input         		| 32x32x1 grayscale image   	|0|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|156|
| ELU					|												|0|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|0|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16			|2416|
| ELU					|												|0|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|0|
| flatten	| input 5x5x16 , output 400       									|0|
| Fully connected		| input 400, output 120       									|48120|
| ELU					|												|0|
| Fully connected		| input 120, output 84       									|10164|
| ELU					|												|0|
| Fully connected		| input 84, output 1      									|85|

The model contains altogether 60,941 trainable parameters. The model deploys Exponential Linear Unit(ELU) activation function instead of ReLU like LeNet. It would keep the negative values with a certain weightage and use them to generate final steering angle.

LeNet has originally been deployed to classify hand written digits on gray scale images. In our problem we need to extract road curvature information which is also represented by lines. Moreover LeNet network is big enough for optimal feature extraction but not too big to overfit for limited amount of training data.






####3. Creation of the Training Set & Training Process

First the network was only trained with the center camera and it was soon realized that the car would drive straight and go off road after the bridge. The reason was lack of turn data. Data from both left and right cam were includied in the training data. All the images were flipped horizontally and also used for training. All together the final dataset was 48216 images. It was further divided into training data set of 38568 images and validation set of 9648 images.

Generator function was used to read the images while training. Model was trained for 5 epochs. Model was saved at the end of every epoch using Keras callback function.

drive.py was modified for the image preprocessing steps so that the data is of the same shape and type which was used to train the network. All the saved models were deployed in the simulator and performance was observed. First epoch generated the best result based on visualization in the simulator. Multiple laps were completed using the generated model. Video was recorded as well using screen capture software Camstudio. Video available at youtube [Youtube Link](https://www.youtube.com/watch?v=JVeGO6B9Reo). Dashcam video was also generated using video.py provided by Udacity.
