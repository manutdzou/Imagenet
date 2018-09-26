# Imagenet
Multi-GPU to train Imagenet by different models

# First download the Imagenet dataset using the official script and transform to TFRecord

like 
![1](/img/1.png)
![2](/img/2.png)

# Train the model using "inference.py"

Here we use 4 gpu cards to train 1000K steps, and the training summary is like this 

![3](/img/3.png)
![4](/img/4.png)
![5](/img/5.png)

The training learning rate is show like this 

![6](/img/6.png)

# Test the model 

Here we use a single gpu and single image crop the top5 accuracy is 0.9048


![7](/img/7.png)
