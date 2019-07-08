import tensorflow as tf 
from tensorflow import keras 
import numpy as np # pip install numpy==1.16.1

# Text classification with movie reviews 
data = keras.datasets.imdb 

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) 

#print(train_data[0]) # integers encoded 

word_index = data.get_word_index() # dictionary tuple for mapping 

word_index = {k:(v+3) for k,v in word_index.items()} 
word_index['<PAD>'] = 0 
word_index['<START>'] = 1 
word_index['<UNK>'] = 2 # unknown 
word_index['<UNUSED>'] = 3 

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 



train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=250) 
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=250) 

#print(len(train_data), len(test_data))

def decode_review(text): 
	return " ".join([reverse_word_index.get(i, "?") for i in text]) 

#print(decode_review(test_data[0])) # decode integers into texts 
#print(len(test_data[0]), len(test_data[1])) 

# model down here 

model = keras.Sequential() 
model.add(keras.layers.Embedding(88000,16)) # 88000: vocab size(every single number represents a word <- vectors!) 
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu')) # activation function 
model.add(keras.layers.Dense(1, activation='sigmoid')) # hidden layer // squash every value between 0 and 1 

model.summary() 

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 

# training data into two sets 
x_val = train_data[:10000] # validation data: we can check how model works(testing accuracy) <- take 10000 data as validation ones 
x_train = train_data[10000:]

y_val = train_labels[:10000] # don't touch any of test data! 
y_train = train_labels[10000:] 

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1) # batch size: how many loaded each time 

results = model.evaluate(test_data, test_labels) 

print(results) 

'''
test_review = test_data[0] 
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0])) 
print(results) 
'''

# Saving and Loading Models 
model.save("model.h5") # save the model as a binary file 

model = keras.models.load_model("model.h5") # do not need to run "modeling" codes once again! 
