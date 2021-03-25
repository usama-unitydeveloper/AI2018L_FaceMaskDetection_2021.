# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initializing learning rate (eeta) batch size and epochs
initial_Lrate = 1e-4
EPOCHS = 17
Batch_Size = 30
#load all images from data set into ARRAY for processing

categories = ["with_mask","without_mask"]
dataset = r"I:\GITHub\Myrepo\FacemaskDetection\dataset"


img_data = []
labels = []

for category in categories:
    path = os.path.join(dataset, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	img_data.append(image)
    	labels.append(category)

# converting labels to binary data
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# convert image data and labels to numpy array for processing


img_data = np.array(img_data, dtype="float32")
labels = np.array(labels)


#spliting of data into training and testing data with 80% for training and 20% for testing
(trainX, testX, trainY, testY) = train_test_split(img_data, labels,test_size=0.20, random_state=42)

# generating different images with different properties
augument = ImageDataGenerator(rotation_range=15,	zoom_range=0.12, shear_range=0.12, width_shift_range=0.2,height_shift_range=0.2,
	fill_mode="nearest")

# load mobileNet for training


Bmodel = MobileNetV2(weights="imagenet",input_tensor=Input(shape=(224, 224, 3)), include_top=False)


# head model that will be placed on the top of base model

Hmodel = Bmodel.output

Hmodel = AveragePooling2D(pool_size=(7, 7))(Hmodel
											)
Hmodel = Flatten(name="flatten")(Hmodel)

Hmodel = Dense(128, activation="relu")(Hmodel)

Hmodel = Dropout(0.5)(Hmodel)

Hmodel = Dense(2, activation="softmax")(Hmodel)

# training model with base model as input and head model as output this is for placing head model on top of base model
Tmodel = Model(inputs=Bmodel.input, outputs=Hmodel)

# Freezing base model layers as these are for training
for layer in Bmodel.layers:
	layer.trainable = False

# compiling our model

print("compiling model...")

opt = Adam(lr=initial_Lrate, decay=initial_Lrate / EPOCHS)

Tmodel.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the head
print(" training head...")

H = Tmodel.fit(augument.flow(trainX, trainY, batch_size=Batch_Size),steps_per_epoch=len(trainX) // Batch_Size,
	validation_data=(testX, testY),validation_steps=len(testX) // Batch_Size,epochs=EPOCHS)

# making predictions on the testing set of given test
print("evaluating network...")


predictIndex = Tmodel.predict(testX, batch_size=Batch_Size)

# find the index with max probuibility

predict = np.argmax(predictIndex, axis=1)

# classification report
classification_report  = classification_report(testY.argmax(axis=1), predictIndex,target_names=lb.classes_)
print(classification_report)

# serialize the model to disk
print("saving mask detector model...")
Tmodel.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")