#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_cell_magic('time', '', 'import keras\nfrom keras.datasets import cifar10')


# In[13]:


cifar10_class_names ={
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Boat",
    9: "Truck"
}


# In[14]:


#Load the entire dataset
(x_train, y_train), (x_test, y_test) =cifar10.load_data()


# In[15]:


for i in range(1000): #Loop through each picture in the dataset
    sample_image = x_train[i] # grab an image from the dataset
    image_class_number = y_train[i][0] #grab the image expected class id
    image_class_name = cifar10_class_names[image_class_number]  # look up the class name from the class id
    plt.imshow(sample_image) #Draw the image as a plot
    plt.title(image_class_name)
    plt.show()


# In[17]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path


# In[18]:


# Normalize dataset to 0-to-1 range
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train/255
x_test =x_test/255


# In[19]:


#Convert class vectors to binary class
# our labels are single values from 0-9
#Instead, we want each label to be an array with on element set to 1 

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test,10)


# In[21]:


#Create a model and add layers
model = Sequential()
model.add(Dense(512, activation="relu", input_shape =(32,32,3)))
model.add(Dense(10, activation ='softmax'))


# In[23]:


model.summary()


# In[27]:


#Convolutional Layers

model = Sequential()
#add convolutional layers
model.add(Conv2D(32,(3,3), padding="same",activation="relu", input_shape =(32,32,3)))
model.add(Conv2D(32,(3,3), activation="relu"))

model.add(Conv2D(64,(3,3), padding ="same", activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
 #whenever we want to transform from Convolutional layer to dense layer, we need to take care that we will no longer working with 2D data to do that we create Flaten layers
model.add(Flatten())


model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation ='softmax'))


# In[28]:


model.summary()


# In[29]:


#Max Poolings

model = Sequential()
#add convolutional layers
model.add(Conv2D(32,(3,3), padding="same",activation="relu", input_shape =(32,32,3)))
model.add(Conv2D(32,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2))) # to scale down the results of convolutional model by keeping only large number of its/ most important value. this will increase the efficiency of model and keep data clean and effective


model.add(Conv2D(64,(3,3), padding ="same", activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
 #whenever we want to transform from Convolutional layer to dense layer, we need to take care that we will no longer working with 2D data to do that we create Flaten layers
model.add(Flatten())


model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation ='softmax'))


# In[30]:


model.summary()


# In[31]:


#Drop out

model = Sequential()
#add convolutional layers
model.add(Conv2D(32,(3,3), padding="same",activation="relu", input_shape =(32,32,3)))
model.add(Conv2D(32,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2))) # to scale down the results of convolutional model by keeping only large number of its/ most important value. this will increase the efficiency of model and keep data clean and effective
model.add(Dropout(0.25)) #normally 25% - 50% is work well #drop out use to avoid overfitting

model.add(Conv2D(64,(3,3), padding ="same", activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#Whenever we want to transform from Convolutional layer to dense layer, we need to take care that we will no longer working with 2D data to do that we create Flaten layers
model.add(Flatten())


model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation ='softmax'))


# In[32]:


model.summary()


# In[37]:


#Complete NN for Image Recognition


#Compile the model
model.compile(
    loss="categorical_crossentropy", # cross entropy because the value input 0-9
    optimizer= "adam",
    metrics=['accuracy']
    
)


# In[38]:


# Next we start training process

#Train the model

model.fit( 
    x_train,
    y_train,
    batch_size=32, #ideal is between 32-128
    epochs=30, # how do we define this number?
    validation_data=(x_test,y_test),
    shuffle=True 
)


# In[168]:


#Save neural network Structure

model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)


# In[169]:


#Save neural network's trained weights

model.save_weights("model_weights.h5")


# In[198]:


#Making Prediction

from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np


# In[ ]:





# In[199]:


class_labels =[
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]


# In[200]:


##Load the json file that contains the model's structure
f=Path("model_structure.json")
model_structure = f.read_text()


# In[201]:


#Recreate the Keras model object from the json data
model = model_from_json(model_structure)


# In[202]:


#Re-load the model's trained weights
model.load_weights("model_weights.h5")


# In[211]:


#Load an image file to test, resizing it to 32x32 pixels (as required by this model)
img = image.load_img("car.png", target_size=(32,32))


# In[212]:


#Convert the image to a numpy array
image_to_test = image.img_to_array(img)/255


# In[213]:


#Add a fourth dimension to the image(since Keras expects a list of images, not a single image)

list_of_images = np.expand_dims(image_to_test, axis=0)


# In[214]:


#Make a prediction using the model

results = model.predict(list_of_images)


# In[215]:


#Since we are only testing one image, we only need to check the first results
single_result = results[0]


# In[216]:


#we will get a likelihood score for all 10 possible classes. find out which class had hightest value

most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]


# In[217]:


#Get the name of the most likely class

class_label = class_labels[most_likely_class_index]


# In[218]:


#Print the results
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))


# ## Pretrain Neutral network

# In[219]:


from keras.applications import vgg16


# In[220]:


#load Keras's VGG16 model that was pre-trained against the Imagenet database

model = vgg16.VGG16()


# In[222]:


#Load the image file, resizing it to 224x224 pixels (requied by this model)
img = image.load_img("bay.jpg", target_size=(224,224))


# In[223]:


#Convert the image to a numpy array
x=image.img_to_array(img)


# In[224]:


#Add a fourth dimension (since Keras expects a list of images)

x = np.expand_dims(x, axis=0)


# In[225]:


#Normalize the input image pinxel values to the range used when training the NN
x=vgg16.preprocess_input(x)


# In[226]:


#Run the image through the deep neural network to make a prediction
predictions = model.predict(x)


# In[229]:


#Look up the nanmes in the predicted classes
predicted_classes = vgg16.decode_predictions(predictions, top=9)

print("Top prediction for this image:")


# In[230]:


for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Prediction:{} - {:2f}".format(name, likelihood))


# ### Feature Extraction

# In[231]:


from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16


# In[241]:



import os
import glob
dog = "D:\Training\Weekend\Deep Learning\Ex_Files_Deep_Learning_Image_Recog_Upd (1)\Ex_Files_Deep_Learning_Image_Recog_Upd\Exercise Files\Ch05\training_data\dogs" # Enter Directory of all images 
not_dog = "D:\Training\Weekend\Deep Learning\Ex_Files_Deep_Learning_Image_Recog_Upd (1)\Ex_Files_Deep_Learning_Image_Recog_Upd\Exercise Files\Ch05\training_data\not_dogs"


# In[242]:


#Path to folders with training data

dog_path = dog/"dogs"
not_dog_path = not_dog/"not_dogs"

images =[]
labels =[]


# In[243]:


dog_path = os.path.join(img_dir_dog,'*g')
files_dog = glob.glob(dog_path)

not_dog_path = os.path.join(img_dir_not_dog,'*g')
files_not_dog = glob.glob(not_dog_path)


# In[247]:


#load all the not-dog images

for img in files_not_dog.glob("*.png"):
    img = image.load_img(img) #load image from disk
    image_array = image.img_to_array(img) # convert the image to a numpy array
    images.append(image_array) # add the image to the list of images
    labels.append(0) # for each 'not dog' image, the expected value should be 0


# In[245]:


#load all the dogs images

for img in dog_path.glob("*.png"):
    img = image.load_img(img) #load image from disk
    image_array = image.img_to_array(img) # convert the image to a numpy array
    images.append(image_array) # add the image to the list of images
    labels.append(1) # for each 'dog' image, the expected value should be 1


# In[ ]:


#Create a singple numpy array with all the images we loaded

x_train = np.array(images)


# In[ ]:


#ALso convert the labels to a numpy array

y_train = np.array(labels)


# In[ ]:


# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)


# In[ ]:


#Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape(64,64,3))


# In[ ]:


#Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)


# In[ ]:


#Save the array of extracted features to a file
joblib.dump(features_x, 'x_train.dat')


# In[ ]:


#Save the matching arrray of expected values to a file
joblib.dump(y_train, "y_train.dat")


# ### Training  a new neutral network with extracted features
# 
# 

# In[ ]:


import keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib


# In[ ]:


#Load dataset
x_train = joblib.load("x_train.dat")
y_train = joblib.load('y_train.dat')


# In[ ]:


#create a model and add layers

model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:1]))
model.add(Dense(256, activation='relu'))
model.add(Droput(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


#compile the model

model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy']

)


# In[ ]:


#train the model

model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
    
)


# In[ ]:


#save neural network structure
model_structure = model.to_json()
f=Path("model_structure.json")
f.write_text(model_structure)


# In[ ]:


#save neural networks' trained weights
model.save_weights("model_weights.h5")


# ### Making prediction with transfer learning

# In[ ]:


from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16


# In[ ]:


##Load the json file that contains the model's structure
f=Path("model_structure.json")
model_structure = f.read_text()


# In[ ]:


#Recreate the Keras model object from the json data
model = model_from_json(model_structure)


# In[ ]:


#Re-load the model's trained weights
model.load_weights("model_weights.h5")


# In[ ]:


#Load an image file to test, resizing it to 32x32 pixels (as required by this model)
img = image.load_img("car.png", target_size=(32,32))


# In[ ]:


#Convert the image to a numpy array
image_to_test = image.img_to_array(img)/255


# In[ ]:


#Add a fourth dimension to the image(since Keras expects a list of images, not a single image)

list_of_images = np.expand_dims(image_to_test, axis=0)


# In[ ]:


#normlaize the data
images = vgg16.preprocess_input(images)


# In[ ]:


#use the pre-trained neural network to extract features from our test image

feature_extraction_model = vgg16.VGG16(weights="imagenet", include_top=Falese, input_shapes=(64,64,3))
features = feature_extraction_model.predict(images)    


# In[ ]:


#Given the extracted features make a final prediction using our own model

results = model.predict(features)


# In[ ]:


#Since we are only testing one image with possible class
sing_result = results[0][0]


# In[ ]:


#Print the results

print("Likelihood that this image contains a dog:{}%".format(int(single_result)*100))


# In[ ]:





# In[ ]:




