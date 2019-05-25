from django.shortcuts import render
from django.http import HttpResponse
from django.core.files import File
from FaceRecognition import settings
import os
import pickle
import cv2
import numpy as np

# Create your views here.


def index(request):
    response = HttpResponse()
    response.writelines("day la recognition")
    return response


def convertfileimage(l0):
    file_path = os.path.join(settings.BASE_DIR, 'recognition', l0)
    with open(file_path, 'r') as f:
        s = f.read()
    f.close()
    z = s.strip('_').split('_')
    l = []
    for a in z:
        l.append(int(a))
    file_path2 = os.path.join(settings.BASE_DIR, 'recognition', l0 + '.JPG')
    with open(file_path2, 'wb') as f:
        f.write(bytes(l))
    f.close()
    return file_path2


def knn(train_folder):
    digits = []
    labels = []
    dict_labels = {}
    model = cv2.ml.KNearest_create()

    listfolder = os.listdir(train_folder)
    for i in range(len(listfolder)):
        dict_labels[i] = listfolder[i]
        for a in os.listdir(os.path.join(train_folder, listfolder[i])):
            image = cv2.imread(os.path.join(
                train_folder, listfolder[i], a), cv2.COLOR_BGR2GRAY)
            digits.append(image)
            labels.append(i)

    x = np.array(digits)
    X_train = x[:].reshape(-1, 10000).astype(np.float32)

    # X_train = np.float32(digits)
    y_train = np.asarray(labels)

    # print(X_train)
    model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    return model, dict_labels


def predict(model, dict_labels, image_path):
    imagetest = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    x = np.array(imagetest)
    test_img = x.reshape(-1, 10000).astype(np.float32)
    retval, results, neigh_resp, dists = model.findNearest(test_img, 3)
    return dict_labels[int(retval)]


def index2(request, s=None):
    response = HttpResponse()
    l = s.split('=')
    file_path = os.path.join(settings.BASE_DIR, 'recognition', l[0])
    if (l[1] == '-1'):
        image_path = convertfileimage(l[0])
        model, dict_labels = knn(os.path.join(
            settings.BASE_DIR, 'recognition', 'image'))
        result = predict(model, dict_labels, image_path)
        response.write(result)
        os.remove(file_path)
        os.remove(image_path)
        return response
    else:
        if (os.path.isfile(file_path)):
            with open(file_path, 'a') as f:
                a = f.write(l[1])
            f.close()
        else:
            with open(file_path, 'w') as f:
                a = f.write(l[1])
            f.close()
        response.write(a)
        return response


def classify():
    return 'tung'
