import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 

# https://www.tensorflow.org/alpha/tutorials/keras/basic_classification
data = keras.datasets.fashion_mnist 

(train_images, train_labels), (test_images, test_labels) = data.load_data() 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # match these to indices 

# shrink data size 
train_images = train_images/255.0
test_images = test_images/255.0

#print(train_images[7]) # pixel values(rgb) => need to shrink the data down! 

#plt.imshow(train_images[7], cmap=plt.cm.binary) # cmap=plt.cm.binary: black and white images 
#plt.show() 

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)), # input layer with flattening data 
	keras.layers.Dense(128, activation="relu"), # dense layer with 128 neurons and relu function 
	keras.layers.Dense(10, activation="softmax") # softmax: prob of thinking certain values 
	]) 

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 

model.fit(train_images, train_labels, epochs=5) # train the model 

#test_loss, test_acc = model.evaluate(test_images, test_labels) 

#print("Tested Acc: ", test_acc) 

prediction = model.predict(test_images) 
#prediction = model.predict(test_images[7]) 
#print(prediction) 
#print(prediction[0]) 
#print(np.argmax(prediction[0])) 
for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary) 
	plt.xlabel("Actual: " + class_names[test_labels[i]]) 
	plt.title("Prediction" + class_names[np.argmax(prediction[i])]) # name of items 
	plt.show() 

