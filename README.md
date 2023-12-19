# Download rockpaperscissors dataset
!wget --no-check-certificate \
--2023-12-19 15:40:38--  https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip
Resolving github.com (github.com)... 140.82.114.3
Connecting to github.com (github.com)|140.82.114.3|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://objects.githubusercontent.com/github-production-release-asset-2e65be/391417272/7eb836f2-695b-4a46-9c78-b65867166957?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231219%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231219T153926Z&X-Amz-Expires=300&X-Amz-Signature=c2c8d69b0add27b0ac7dc5f3a77fd11b258d815648b7fdd27c9e272ba5be35b3&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=391417272&response-content-disposition=attachment%3B%20filename%3Drockpaperscissors.zip&response-content-type=application%2Foctet-stream [following]
--2023-12-19 15:40:38--  https://objects.githubusercontent.com/github-production-release-asset-2e65be/391417272/7eb836f2-695b-4a46-9c78-b65867166957?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231219%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231219T153926Z&X-Amz-Expires=300&X-Amz-Signature=c2c8d69b0add27b0ac7dc5f3a77fd11b258d815648b7fdd27c9e272ba5be35b3&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=391417272&response-content-disposition=attachment%3B%20filename%3Drockpaperscissors.zip&response-content-type=application%2Foctet-stream
Resolving objects.githubusercontent.com (objects.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
Connecting to objects.githubusercontent.com (objects.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 322873683 (308M) [application/octet-stream]
Saving to: ‘/tmp/rockpaperscissors.zip’

/tmp/rockpapersciss 100%[===================>] 307.92M   206MB/s    in 1.5s    

2023-12-19 15:40:40 (206 MB/s) - ‘/tmp/rockpaperscissors.zip’ saved [322873683/322873683]

22.41
[3]
import tensorflow as tf
from tensorflow.keras.models import load_model

22.42
[4]
pip install tensorflow
Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.15.0)
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=23.5.26 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (23.5.26)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.5.4)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.9.0)
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (16.0.6)
Requirement already satisfied: ml-dtypes~=0.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.23.5)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.3.0)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (23.2)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (67.7.2)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.5.0)
Requirement already satisfied: wrapt<1.15,>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.14.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.34.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.60.0)
Requirement already satisfied: tensorboard<2.16,>=2.15 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.15.1)
Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.15.0)
Requirement already satisfied: keras<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.15.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.42.0)
Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (2.17.3)
Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (1.2.0)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (3.5.1)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (2.31.0)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow) (3.0.1)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (5.3.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (0.3.0)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (4.9)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow) (1.3.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (2023.11.17)
Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow) (2.1.3)
Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (0.5.1)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow) (3.2.2)
22.43
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
Collecting split-folders
  Downloading split_folders-0.5.1-py3-none-any.whl (8.4 kB)
Installing collected packages: split-folders
Successfully installed split-folders-0.5.1
Copying files: 2188 files [00:00, 2411.91 files/s]
22.47
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
22.48
[12]
!rm -rf /tmp/rockpaperscissors/rps-cv-images/.ipynb_checkpoints
22.48
[13]
# Image Augmentation for duplicating image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
22.49
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
Epoch 1/20
25/25 - 33s - loss: 1.0395 - accuracy: 0.4387 - val_loss: 0.7556 - val_accuracy: 0.7500 - 33s/epoch - 1s/step
Epoch 2/20
25/25 - 32s - loss: 0.4495 - accuracy: 0.8512 - val_loss: 0.2417 - val_accuracy: 0.9125 - 32s/epoch - 1s/step
Epoch 3/20
25/25 - 44s - loss: 0.2574 - accuracy: 0.9125 - val_loss: 0.2588 - val_accuracy: 0.9250 - 44s/epoch - 2s/step
Epoch 4/20
25/25 - 29s - loss: 0.2285 - accuracy: 0.9312 - val_loss: 0.1979 - val_accuracy: 0.9375 - 29s/epoch - 1s/step
Epoch 5/20
25/25 - 29s - loss: 0.2257 - accuracy: 0.9350 - val_loss: 0.1937 - val_accuracy: 0.9312 - 29s/epoch - 1s/step
Epoch 6/20
25/25 - 30s - loss: 0.1651 - accuracy: 0.9538 - val_loss: 0.2158 - val_accuracy: 0.9312 - 30s/epoch - 1s/step
Epoch 7/20
25/25 - 32s - loss: 0.1763 - accuracy: 0.9438 - val_loss: 0.1953 - val_accuracy: 0.9375 - 32s/epoch - 1s/step
Epoch 8/20
25/25 - 30s - loss: 0.1227 - accuracy: 0.9625 - val_loss: 0.1245 - val_accuracy: 0.9500 - 30s/epoch - 1s/step
Epoch 9/20
25/25 - 30s - loss: 0.1366 - accuracy: 0.9563 - val_loss: 0.1707 - val_accuracy: 0.9375 - 30s/epoch - 1s/step
Epoch 10/20
25/25 - 29s - loss: 0.1365 - accuracy: 0.9500 - val_loss: 0.1003 - val_accuracy: 0.9875 - 29s/epoch - 1s/step
Epoch 11/20
25/25 - 28s - loss: 0.0867 - accuracy: 0.9737 - val_loss: 0.1676 - val_accuracy: 0.9438 - 28s/epoch - 1s/step
Epoch 12/20
25/25 - 29s - loss: 0.1484 - accuracy: 0.9425 - val_loss: 0.0385 - val_accuracy: 0.9937 - 29s/epoch - 1s/step
Epoch 13/20
25/25 - 29s - loss: 0.1035 - accuracy: 0.9700 - val_loss: 0.2001 - val_accuracy: 0.9312 - 29s/epoch - 1s/step
Epoch 14/20
25/25 - 34s - loss: 0.1117 - accuracy: 0.9600 - val_loss: 0.1142 - val_accuracy: 0.9750 - 34s/epoch - 1s/step
Epoch 15/20
25/25 - 30s - loss: 0.1093 - accuracy: 0.9600 - val_loss: 0.0646 - val_accuracy: 0.9875 - 30s/epoch - 1s/step
Epoch 16/20
25/25 - 29s - loss: 0.0964 - accuracy: 0.9725 - val_loss: 0.0610 - val_accuracy: 0.9875 - 29s/epoch - 1s/step
Epoch 17/20
25/25 - 29s - loss: 0.0436 - accuracy: 0.9887 - val_loss: 0.0749 - val_accuracy: 0.9875 - 29s/epoch - 1s/step
Epoch 18/20
25/25 - 31s - loss: 0.0750 - accuracy: 0.9787 - val_loss: 0.1122 - val_accuracy: 0.9875 - 31s/epoch - 1s/step
Epoch 19/20
25/25 - 29s - loss: 0.0400 - accuracy: 0.9837 - val_loss: 0.0435 - val_accuracy: 0.9937 - 29s/epoch - 1s/step
Epoch 20/20
25/25 - 29s - loss: 0.0664 - accuracy: 0.9725 - val_loss: 0.0805 - val_accuracy: 0.9937 - 29s/epoch - 1s/step
23.03
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
