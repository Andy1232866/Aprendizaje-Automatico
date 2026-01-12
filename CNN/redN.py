import random
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D, Dropout, LeakyReLU, BatchNormalization

learning_rate_manual = 0.0005

opt = Adam(learning_rate = learning_rate_manual)

# ----- CARGA DE DATOS PROCESADOS -----
X_train = np.load('npy_files/X_train.npy')
y_train = np.load('npy_files/y_train.npy')
X_val = np.load('npy_files/X_val.npy')
y_val = np.load('npy_files/y_val.npy')
X_test = np.load('npy_files/X_test.npy')
y_test = np.load('npy_files/y_test.npy')

# ----- DATOS PARA LA VISUALIZACIÓN -----
X_trainO = np.load('npy_files/X_trainO.npy')
y_trainL = np.load('npy_files/y_trainL.npy')
X_testO = np.load('npy_files/X_testO.npy')
y_testL = np.load('npy_files/y_testL.npy')
labels_dict = np.load('npy_files/labels_dict.npy', allow_pickle = True).item()
class_names = list(labels_dict)

print("\nDATOS CARGADOS\n")

# ----- VISUALIZACIÓN DE UNA IMAGEN DE CADA CLASE -----
unicaL = np.unique(y_testL)

plt.figure(figsize=(12, 6))

for i, label in enumerate(unicaL):
    indices = np.where(y_testL == label)[0]
    
    idx = random.choice(indices)
    img = X_testO[idx]
    
    plt.subplot(1, len(unicaL), i + 1)
    plt.imshow(img)
    plt.title(f"Clase: {class_names[label]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ----- DEFINICIÓN DEL MODELO -----
model = Sequential([
    ZeroPadding2D(padding = (1, 1), input_shape = (128, 128, 3), name = "zeropa"),
    Conv2D(32, (3, 3), name = 'conv1'),
    BatchNormalization(),
    LeakyReLU(alpha = 0.1),
    MaxPooling2D(2, 2, name = 'pool1'),

    Conv2D(64, (3, 3), name = 'conv2'),
    BatchNormalization(),
    LeakyReLU(alpha = 0.1),
    MaxPooling2D(2, 2, name = 'pool2'),

    Conv2D(128, (3, 3), name = 'conv3'),
    BatchNormalization(),
    LeakyReLU(alpha = 0.1),
    MaxPooling2D(2, 2, name = 'pool3'),
    
    Conv2D(256, (3, 3), name = 'conv4'),
    BatchNormalization(),
    LeakyReLU(alpha = 0.1),
    MaxPooling2D(2, 2, name = 'pool4'),
    
    Flatten(name='flatten'),
    Dense(64, name = 'dens1'),
    LeakyReLU(alpha = 0.5),
    Dropout(0.15),
    Dense(16, name = 'dens1'),
    LeakyReLU(alpha = 0.5),
    Dropout(0.10),
    Dense(4, activation = 'softmax', name = 'output')
])

early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 20, batch_size = 64, validation_data = (X_val, y_val), callbacks = [early_stop])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"La precisión del modelo es de: {test_accuracy * 100:.2f}%")
print(f"La pérdida del modelo es de: {test_loss:.4f}%")

# ----- VISUALIZACIÓN DE PREDICCIONES ALEATORIAS -----
num_rows, num_cols = 5, 3
num_images = num_rows * num_cols

indices = np.random.choice(len(X_testO), num_images, replace = False)

plt.figure(figsize = (12, 16))

for i, idx in enumerate(indices):   
    img = X_testO[idx]
    true_label = y_testL[idx]
    
    pred = model.predict(np.expand_dims(X_test[idx], axis = 0), verbose = 0)
    pred_label = np.argmax(pred)
    
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(img)
    
    color = "green" if pred_label == true_label else "red"
    
    plt.title(f"Real: {class_names[true_label]}\nPred: {class_names[pred_label]}", color = color, fontsize = 10)
    plt.axis("off")

plt.tight_layout()
plt.show()

# ----- GRÁFICOS DE EXACTITUD Y PÉRDIDA -----
plt.figure(figsize = (12, 10))
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylim(0, 1)
plt.title('Modelo Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Épocas')
plt.legend(['train', 'valid'], loc = 'lower right')
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim([0, 2])
plt.title('Modelo Loss')
plt.ylabel('Loss')
plt.xlabel('Épocas')
plt.legend(['train', 'valid'], loc = 'upper right')

# ----- MATRIZ DE CONFUSIÓN -----
y_pred = model.predict(X_test)
y_predC = np.argmax(y_pred, axis = 1)
y_trueC = np.argmax(y_test, axis = 1)
cm = confusion_matrix(y_trueC, y_predC)

plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = class_names, yticklabels = class_names)

plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()

# ----- VISUALIZACIÓN DE LAS CAPAS DE CONVOLUCIÓN -----
img = X_test[0]

layer_outputs = [layer.output for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]
layer_names = [layer.name for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]
activation_model = Model(inputs = model.layers[0].input, outputs = layer_outputs)

feature_maps = activation_model.predict(np.expand_dims(img, axis = 0))

n_layers = len(layer_names)
plt.figure(figsize = (20, 3 * n_layers))

for idx, (layer_name, feature_map) in enumerate(zip(layer_names, feature_maps)):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]

        n_cols = min(n_features, 8)
        display_grid = np.zeros((size, size * n_cols))

        for i in range(n_cols):
            epsilon = 1e-7
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= (x.std() + epsilon)
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size:(i + 1) * size] = x

        plt.subplot(n_layers, 1, idx + 1)
        plt.title(layer_name, fontsize = 10)
        plt.grid(False)
        plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
        plt.axis('off')

plt.tight_layout()
plt.show()