import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from numpy.random import RandomState
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tqdm import tqdm

def preprocess(data, test = False):
    labels = []
    if not test:
        labels = data['label']
        pixels = data.drop('label', axis = 1)
    else:
        pixels = data
    images = []
    for x in tqdm(range(0, len(pixels))):
        tp = np.array(pixels.iloc[x:x+1, :])
        tp = tp / 255
        tp = tp.reshape(28, 28)
        images.append(tp)
    images = np.array(images)
    if test: return images
    return labels, images
    
dataset = pd.read_csv("/content/train.csv")
testset = pd.read_csv("/content/test.csv")
rng = RandomState()
train = dataset.sample(frac=0.8, random_state=rng)
test = dataset.loc[~dataset.index.isin(train.index)]

train_labels, train_imgs = preprocess(train)
test_labels, test_imgs = preprocess(test)

ftest_images = preprocess(testset, True)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_imgs, train_labels, epochs=10, 
                    validation_data=(test_imgs, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_imgs, test_labels, verbose=2)
flabels = model.predict(ftest_images)

test_preds=[]
for i in tqdm(range(len(flabels))):
    test_preds.append(np.argmax(flabels[i]))

pred_df = pd.DataFrame({"ImageId":testset.index+1, "Label":test_preds})
pred_df.to_csv("submission.csv", index = False)
