from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers  import Dropout
from tensorflow.keras.layers  import Dense
from tensorflow.keras.layers  import Flatten
from tensorflow.keras.layers  import Softmax
from tensorflow.keras.layers  import Input
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

data =[]
labels = []
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

directory = os.path.dirname(__file__)
CATERGORIES = ['with_mask','without_mask']

for cateogory in CATERGORIES:
    path = os.path.join(directory,'dataset/'+cateogory)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(cateogory)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data,dtype='float32')
labels = np.array(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.2,random_state=32,stratify=labels)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,
                         horizontal_flip=True,fill_mode='nearest')
baseModel = MobileNetV2(weights='imagenet', include_top=False,input_tensor=Input(shape=(224,224,3)))

headmodel = baseModel.output
headmodel = AvgPool2D(pool_size=(7,7))(headmodel)
headmodel = Flatten(name='flattern')(headmodel)
headmodel = Dense(128,activation='relu')(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(2,activation='softmax')(headmodel)

model = Model(inputs=baseModel.input,outputs=headmodel)

for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(learning_rate=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

H = model.fit(aug.flow(trainX,trainY,batch_size=BS),steps_per_epoch=len(trainX)//BS,
              validation_data=(testX,testY),validation_steps=len(testX)//BS, epochs=EPOCHS)

preIdxs = model.predict(testX,batch_size=BS)
preIdxs = np.argmax(preIdxs,axis=1)
print(classification_report(testY.argmax(axis=1),preIdxs,target_names=lb.classes_))
#model.save("mask_detector.model",save_format="h5")

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

