from random import choice
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from skimage import io
import face_alignment
import os
import h5
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import keras
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
import cv2  # opencv
import pandas as pd
import numpy as np  # Linear algebra
import mtcnn
import logging
import time
from turtle import color

from matplotlib.patches import Circle, Rectangle
from matplotlib.pyplot import hist
from networkx.algorithms.tree import recognition
from scipy.constants import lb
from scipy.spatial import distance
from tensorflow._api.v2 import math
from tensorflow.python.ops.summary_ops_v2 import graph

logging.getLogger('tensorflow').disabled = True


# print version
print(mtcnn.__version__)


print(os.listdir("C:/Users/Jawabreh/MFR/Test1"))

img = cv2.imread('C:/Users/Jawabreh/MFR/Test1/Dataset/1.jpg')
# img = cv2.imread('C:/Users/hossam/RMFD/test/lichen')

plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt-yticks([]) # to hide tick values on X and Y axis

plt.show()


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    print(type(results))
    print(results)

    # fin
    if results:
        x1, y1, width, height = results[0]['box']
    else:
        x1, y1, width, height = [0, 0, 128, 128]

    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array
# draw an image with detected objects


def draw_image_with_boxes(filename, result_list):
    # load the image
    data = plt.imread(filename)
# plot the image
    plt.imshow(data)
# get the context for drawing boxes
    ax = plt.gca()
# plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
# create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
# draw the box
        ax.add_patch(rect)
# draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot
    plt.show()


filename = 'C:/Users/Jawabreh/MFR/Ahmad_Jawbreh_pic.jpg'
# load image from file
pixels = plt.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)
# load the photo and extract the face
# pixels = extract_face("C:/Users/Jawabreh/MFR/tt/12.jpg")
pixels = extract_face('C:/Users/Jawabreh/MFR/0_0_0.jpg')
plt.imshow(pixels)
plt.show()
print(pixels.shape)


def load_face(dir):
    faces = list()

    # enumerate files

    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)

    return faces


def load_dataset(dir):
    # list for faces and labels

    X, y = list(), list()

    for subdir in os.listdir(dir):
        path = dir + '/' + subdir + '/'
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" %
              (len(faces), subdir))  # print progress
        X.extend(faces)
        y.extend(labels)

    return np.asarray(X), np.asarray(y)


# load train dataset
#trainX, trainy = load_dataset ('C:/Users/Jawabreh/MFR/trainSet' )
#print (trainX.shape, trainy.shape)

# load test dataset
#testX, testy = load_dataset('C:/Users/Jawabreh/MFR/val')
#print(testX.shape, testy.shape)

# save and compress the dataset for further use
#np.savez_compressed ('RR_masked3.npz',trainX, trainy, testX, testy)

# load face dataset
data = np.load('RRRRRR_masked3.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)


model2 = load_model('C:/Users/Jawabreh/MFR/MyTrainingModel.h5')

print('loaded model')
model2.summary()


def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get
    yhat = model.predict(sample)
    return yhat[0]


# convert each face in the train set into embedding
emdTrainX = list()
for face in trainX:
    emd = get_embedding(model2, face)
    emdTrainX.append(emd)

emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)

# convert each face in the test set into embedding
emdTestX = list()
for face in testX:
    emd = get_embedding(model2, face)
    emdTestX. append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)

# save arrays to one file in compressed format
np.savez_compressed('RRRRRR_masked4.npz', emdTrainX, trainy, emdTestX, testy)


print("Dataset: train=%d, test=%d" % (emdTrainX.shape[0], emdTestX.shape[0]))
# normalize input vectors

in_encoder = Normalizer(norm='l2')

emdTrainX_norm = in_encoder.transform(emdTrainX)

emdTestX_norm = in_encoder.transform(emdTestX)

# label encode targets

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy_enc = out_encoder.transform(trainy)
testy_enc = out_encoder.transform(testy)

# fit model

#model = SVC(kernel='sigmoid', probability=True)
#model.fit(emdTrainX_norm, trainy_enc)
model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
model.fit(emdTrainX_norm, trainy_enc)
# save the classifier as pickle file
with open('mymodel.pkl', 'wb') as f:
    pickle.dump(model, f)

# predict

yhat_train = model.predict(emdTrainX_norm)
yhat_test = model.predict(emdTestX_norm)

# score

score_train = accuracy_score(trainy_enc, yhat_train)
score_test = accuracy_score(testy_enc, yhat_test)
# evaluation
loss_train = hamming_loss(trainy_enc, yhat_train)
loss_test = hamming_loss(testy_enc, yhat_test)

# summarize
print('loss: train=%.3f, test=%.3f' % (loss_train*100, loss_test*100))
print('Accuracy: train=%.3f,test=%.3f' % (score_train*100, score_test*100))


print(confusion_matrix(testy_enc, yhat_test))
print(classification_report(testy_enc, yhat_test))
print('-----------------------------------------------------------------------')
print(confusion_matrix(trainy_enc, yhat_train))
print(classification_report(trainy_enc, yhat_train))


# select a random face from test set

selection = choice([i for i in range(testX.shape[0])])

random_face = testX[selection]

random_face_emd = emdTestX_norm[selection]

random_face_class = testy_enc[selection]

random_face_name = out_encoder.inverse_transform([random_face_class])


# prediction for the face
print(type(random_face_emd))

print(random_face_emd.shape)

samples = np.expand_dims(random_face_emd, axis=0)

print(samples.shape)

yhat_class = model.predict(samples)

yhat_prob = model.predict_proba(samples)

# get name

class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index]*100

predict_names = out_encoder.inverse_transform(yhat_class)
all_names = out_encoder.inverse_transform([0, 1, 2, 3, 4])
#print('predicted:%s (%.3f)' % (predict_names[0], class_probability))
print('predicted:\n%s \n%s' % (all_names, yhat_prob[0] * 100))
print('Expected:%s' % random_face_name[0])
# plot face
plt.imshow(random_face)
title = '%s (%.3f)' % (predict_names[0], class_probability)
plt.title(title)
plt.show()


def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]


# select  a random face from test set
# normalize input vectors
in_encoder = Normalizer()

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)

random_face = extract_face('C:/Users/Jawabreh/MFR/test/0_0_1.jpg')
emdFace = get_embedding(model2, random_face)
emdFace = emdFace.reshape(1, -1)
random_face_emd = in_encoder.transform(emdFace)
# load it
with open('mymodel.pkl', 'rb') as f:
    mymodel = pickle.load(f)

# prediction for the face

samples = random_face_emd

yhat_class = mymodel.predict(samples)

yhat_prob = mymodel.predict_proba(samples)

# get name

class_index = yhat_class[0]

class_probability = yhat_prob[0, class_index]*100

predict_names = out_encoder.inverse_transform(yhat_class)

all_names = out_encoder.inverse_transform(np.arange(len(yhat_prob[0])))

i = 0
for name in all_names:

    print(name, yhat_prob[0][i]*100)
    i = i + 1

# plot face
plt.imshow(random_face)
if (class_probability < 10):
    # print("insideif")
    title = 'Unknown Face'
else:
    # print("inside if")
    title = 'predicted: %s          Expected: %s' % (
        predict_names[0], predict_names[0])
plt.title(title)
plt.show()
