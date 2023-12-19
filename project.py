
# Download rockpaperscissors dataset
!wget --no-check-certificate \
--2023-12-19 15:40:38--  https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip

22.41
[3]
import tensorflow as tf
from tensorflow.keras.models import load_model

22.42
[4]
pip install tensorflow


[5]
import tensorflow as tf


22.46
[6]
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


22.46
[7]
# Extract files
import zipfile, os
22.46
[8]
# Split train set (60%) and validation set (40%)
!pip install split-folders



[9]
# Set train and validation directory for each rock, paper, scissors
image_dir = '/tmp/rockpaperscissors/image'
22.47
[10]
# Count the number of train and validation images
train_set = (
Train Set      : 1312
Validation Set : 876
22.48
[11]
train_dir      = os.path.join(image_dir, 'train')
validation_dir = os.path.join(image_dir, 'val')
['paper', 'rock', 'scissors']
['paper', 'rock', 'scissors']



[12]
!rm -rf /tmp/rockpaperscissors/rps-cv-images/.ipynb_checkpoints


[13]
# Image Augmentation for duplicating image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


[14]
# Prepare the training and validation data with .flow_from_directory()
train_generator = train_datagen.flow_from_directory(
Found 1312 images belonging to 3 classes.
Found 876 images belonging to 3 classes.
22.49
[15]
# Image Augmentation for duplicating image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
22.49
[16]
# Build the model with Convolutional Neural Network (CNN) and MaxPooling
import tensorflow as tf
22.49
[17]
# Build the model with Convolutional Neural Network (CNN) and MaxPooling
import tensorflow as tf
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_6 (Conv2D)           (None, 78, 118, 32)       896       
                                                                 
 max_pooling2d_6 (MaxPoolin  (None, 39, 59, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_7 (Conv2D)           (None, 37, 57, 64)        18496     
                                                                 
 max_pooling2d_7 (MaxPoolin  (None, 18, 28, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_8 (Conv2D)           (None, 16, 26, 128)       73856     
                                                                 
 max_pooling2d_8 (MaxPoolin  (None, 8, 13, 128)        0         
 g2D)                                                            
                                                                 
 conv2d_9 (Conv2D)           (None, 6, 11, 512)        590336    
                                                                 
 max_pooling2d_9 (MaxPoolin  (None, 3, 5, 512)         0         
 g2D)                                                            
                                                                 
 flatten_2 (Flatten)         (None, 7680)              0         
                                                                 
 dropout_1 (Dropout)         (None, 7680)              0         
                                                                 
 dense_3 (Dense)             (None, 512)               3932672   
                                                                 
 dense_4 (Dense)             (None, 3)                 1539      
                                                                 
=================================================================
Total params: 4617795 (17.62 MB)
Trainable params: 4617795 (17.62 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
22.50
[18]
# Compile the model with 'categorical_crossentropy' loss function and Adam optimimzer
model.compile(
22.50
[19]
# Create TensorBoard
%load_ext tensorboard
22.50
[20]
# Create TensorBoard
%load_ext tensorboard
The tensorboard extension is already loaded. To reload it, use:
  %reload_ext tensorboard


[21]
# Train the model with model.fit()
history = model.fit(
23.15
[22]
# Visualize accuracy and loss plot
import matplotlib.pyplot as plt
23.15
[23]
# Predicting Image
import numpy as np
23.15
[24]
# Visualize accuracy and loss plot
import matplotlib.pyplot as plt

23.16
[25]
# Predicting Image
import numpy as np

23.16
[26]
# Predicting Image
import numpy as np
