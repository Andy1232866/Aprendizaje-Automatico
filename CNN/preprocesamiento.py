import glob
import cv2 as cv
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

glioma_train = glob.glob('data/Training/glioma/*.jpg')
meningioma_train = glob.glob('data/Training/meningioma/*.jpg')
notumor_train = glob.glob('data/Training/notumor/*.jpg')
pituitary_train = glob.glob('data/Training/pituitary/*.jpg')

plt.figure(figsize = (25,15))

plt.subplot(1, 4, 1)
img = cv.imread(glioma_train[random.randint(len(glioma_train))])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap = 'binary')

plt.subplot(1, 4, 2)
img = cv.imread(meningioma_train[random.randint(len(meningioma_train))])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap = 'binary')

plt.subplot(1, 4, 3)
img = cv.imread(notumor_train[random.randint(len(notumor_train))])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap = 'binary')

plt.subplot(1, 4, 4)
img = cv.imread(pituitary_train[random.randint(len(pituitary_train))])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap = 'binary')

plt.show()

print(len(glioma_train))
print(len(meningioma_train))
print(len(notumor_train))
print(len(pituitary_train))

data_dict_train = {
    'glioma': glioma_train,
    'meningioma': meningioma_train,
    'notumor': notumor_train,
    'pituitary': pituitary_train
}

labels_dict_train = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'pituitary': 3
}

data_dict_train.items()

X_train = []
y_train = []

for label, lista_imagenes in data_dict_train.items():
    print(label)
    for image in lista_imagenes:
        img = cv.imread(image)
        img_resize = cv.resize(img, (128, 128))
        X_train.append(img_resize)
        y_train.append(labels_dict_train[label])

X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)
print(y_train.shape)


glioma_test = glob.glob('data/Testing/glioma/*.jpg')
meningioma_test = glob.glob('data/Testing/meningioma/*.jpg')
notumor_test = glob.glob('data/Testing/notumor/*.jpg')
pituitary_test = glob.glob('data/Testing/pituitary/*.jpg')

plt.figure(figsize = (25,15))

plt.subplot(1, 4, 1)
img = cv.imread(glioma_test[random.randint(len(glioma_test))])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap = 'binary')

plt.subplot(1, 4, 2)
img = cv.imread(meningioma_test[random.randint(len(meningioma_test))])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap = 'binary')

plt.subplot(1, 4, 3)
img = cv.imread(notumor_test[random.randint(len(notumor_test))])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap = 'binary')

plt.subplot(1, 4, 4)
img = cv.imread(pituitary_test[random.randint(len(pituitary_test))])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap = 'binary')

plt.show()

print(len(glioma_test))
print(len(meningioma_test))
print(len(notumor_test))
print(len(pituitary_test))

data_dict_test = {
    'glioma': glioma_test,
    'meningioma': meningioma_test,
    'notumor': notumor_test,
    'pituitary': pituitary_test
}

labels_dict_test = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'pituitary': 3
}

data_dict_test.items()

X_test = []
y_test = []

for label, lista_imagenes in data_dict_test.items():
    print(label)
    for image in lista_imagenes:
        img = cv.imread(image)
        img_resize = cv.resize(img, (128, 128))
        X_test.append(img_resize)
        y_test.append(labels_dict_test[label])

X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_test.shape)
print(y_test.shape)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

X_trainF, X_val, y_trainL, y_valL = train_test_split(X_train_scaled, y_train, random_state = 0, stratify = y_train)

y_trainC = to_categorical(y_trainL, num_classes = 4)
y_valC = to_categorical(y_valL, num_classes = 4)
y_testC = to_categorical(y_test, num_classes = 4)

print("--- FORMAS FINALES ---")
print("Entrenamiento (imágenes):", X_trainF.shape)
print("Entrenamiento (etiquetas):", y_trainC.shape)
print("Validación (imágenes):", X_val.shape)
print("Validación (etiquetas):", y_valC.shape)
print("Prueba (imágenes):", X_test_scaled.shape)
print("Prueba (etiquetas):", y_testC.shape)

np.save('npy_files/X_train.npy', X_trainF)
np.save('npy_files/y_train.npy', y_trainC)

np.save('npy_files/X_test.npy', X_test_scaled)
np.save('npy_files/y_test.npy', y_testC)

np.save('npy_files/X_val.npy', X_val)
np.save('npy_files/y_val.npy', y_valC)

# DATOS PARA LA VISUALIZACIÓN

np.save('npy_files/X_trainO.npy', X_train)
np.save('npy_files/y_trainL.npy', y_train)
np.save('npy_files/X_testO.npy', X_test)
np.save('npy_files/y_testL.npy', y_test)
np.save('npy_files/labels_dict.npy', labels_dict_train)

print("Archivos guardados") # X_train = 4284 (61%), X_test = 1311 (19%), X_val = 1428 (20%)