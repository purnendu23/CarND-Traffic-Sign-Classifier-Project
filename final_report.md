# **Traffic Sign Recognition**

---

[//]: # (Image References)

[image1]: ./images/Training_distribution.png "Training Dataset barplot"
[image2]: ./images/Validation_distribution.png "Validation Dataset barplot"
[image3]: ./images/Test_distribution.png "Test Dataset barplot"

[image4]: ./images/Training_augmented_distribution.png "Augmented training dataset barplot"
[image5]: ./images/Validation_augmented_distribution.png "Augmented validation dataset barplot"

[image6]: ./new_images/img1.png
[image7]: ./new_images/img2.png
[image8]: ./new_images/img3.png
[image9]: ./new_images/img4.png
[image10]: ./new_images/img5.png

---
### Submission Files/Writeup

You're reading the writeup! and here is a link to my [project code](https://github.com/purnendu23/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---

### Data Set Summary & Exploration

##### 1. Basic summary of the data set.
I used the simple python function `len()` to get all this information. Here is an example: `n_train = len(X_train)`. The rest are similar.

* The size of training set is ? **34799**
* The size of the validation set is ? **4410**
* The size of test set is ? **12630**
* The shape of a traffic sign image is ? **(32, 32, 3)**
* The number of unique classes/labels in the data set is ? **43**

##### 2. Exploratory visualization

I define a function `show_distribution()` which is used here and later in the project as well to visualize the dataset at hand. 

![image1] ![image2] ![image3]

You can see from the barplots that the distribution of examples across the 43 sign-classes is skewed. I fix this next by augmenting more data.

---

### Design and Test a Model Architecture

##### 1. Preprocessing

I pre-process the data with the following steps:
*  Conversion to gray-scale

This was done by just taking the average of RGB values. Here is a sample code: 
`X_train = np.sum(X_train/3, axis=3, keepdims=True)`

*  Augmentation of artifitial data to get a more uniform distribution of examples in training and validation set

The first step in augmenting the dataset was combining the training and validation set. My goal was to get the count of examples associated with each class to approximately _"most number number of examples across all classes + 1000 "_. You can see this in the code: `max_count = max(count) + 1000`.
I use three techniques for geometric transformations of images : 'tanslation', 'rotation' and 'perspective-transformation' (gives a top-view transformation of the image). In the project you will find `translation_image(img, translation_lbound, translation_hbound)`, `rotation_image(img, rotation_lbound, rotation_hbound )`, and `perspectiveTransformation_image(img)` which are called for each image. The resuting images are added to the final list along with the original images to get an augmented list of images.
I then use `train_test_split` to split this set into training and validation. The following barplots show the distribution of examples in the 43 different classes of signs.

![Training Augmented][image4] 
![Validation Augmented][image5]


*  Normalization of data

Finally, datasets are normalized. Example code: `X_train_normalized = (X_train-128)/128`

---

##### 2. Model Architecture

1. I start by using the LeNet architecture as described in the _Lesson# 8: Convolutional Neural Network_. This model could not give the minimum accuracy level of .93 for the validation dataset. However, instead of changing the model I focused on using other data-preprocesing techniques (especially data augmentation) as recommended in the project description. I am able to get 97% accuracy on the validation set with the simplest LeNet model. I am quite certain that I can improve this metrics by evolving this model which will be my next step. For now however, I report my approach and results with basic LeNet model.

2. The model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution     	| Input = 32x32x1 Output = 28x28x6 	|
| RELU					|												|
| Pooling  | Input = 28x28x6 Output = 14x14x6 |
| Convolution	    | Output = 10x10x16      									|
| RELU		|         									|
| Pooling				| Input = 10x10x16 Output = 5x5x16 |
|	Flatten    | Input = 5x5x16 Output = 400 |					
|	Fully Connected | Input = 400 Output = 120	|
| RELU |  |
| Fully Connected | Input = 120 Output = 84|
| RELU | |
| Fully Connected | Input = 84 Output = 43|
 
There are only two changes made to the LeNet architecture. First, the input format of the images is changed from (32, 32, 3) to (32, 32, 1) and the final fully connected layer has 43 output values instead of one (because we have 43 different classes)

##### 3. Model Training
I used the basic LeNet architecture. However, I got the required accuracy by data preprocessing and turning the knobs on different hyperparameters. The _number of epochs_, _batch-size_ and _learning rate_ were the main ones.
Fixing the learning rate to 0.00097 gave a gradually increasing accuracy on the validation set. I then experimented with the number of epochs and batch-size to finally fix those at 35 and 156. I used mean of softmax cross entropy to calculate the loss and Adam Optimizer in the backpropogation step.
```
# Calculate Loss/Cost
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

#For running Backprop
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```
##### 4. Solution Approach
I used the LeNet architecture and achieved the desired level of accuracy by thoroughly exploring all the data-preprocessing techniques mentioned in the project description.

My final model results were:
* validation set accuracy of 96.9%
* test set accuracy of ? 90.6%
 

### Test a Model on New Images

##### 1. Acquiring New Images
Here are five German traffic signs that I found on the web:

![Right-of-way at the next intersection][image6]
![Speed limit (30 km/h)][image7]
![Speed limit (60 km/h)][image8] 
![General caution][image9] 
![Road work][image10]


##### 2. Performance on New Images


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection| Right-of-way at the next intersection   									| 
| Speed limit (30 km/h)| Speed limit (30 km/h|
| Speed limit (60 km/h)| Speed limit (60 km/h|
| General caution| General caution|
| Road work| Children crossing|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The test set accuracy was slightly better at 90.6%

##### 3. Model Certainty - Softmax Probabilities

The code for making predictions on my final model (reports the validation accuracy) is located in the 11th cell of the Ipython notebook.
For all images the certainity of the model is almost 100%. That means it is very certain even if it is the wrong prediction.
For example in case of sign - Road Work it is 99.9993% certain that the sign is _Children crossing_ and only 0.0007% certain that it is a road work sign.

| Percentage        	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| General caution | 
| 99.9765 | Speed limit (60km/h)	|
| 99.8264 | Speed limit (30km/h)|
| 0.0007 | Road work|
| 99.0 | Right-of-way at the next intersection|


