import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_seed(28)

#training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

#target_data = np.array([[0], [1], [1], [1]], "float32")

training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

target_data = np.array([[0], [1], [1], [0]], "float32")

model = Sequential()

model.add(Dense(16, input_dim = 2, activation = "sigmoid", name = "capa1"))
model.add(Dense(8, activation = "linear", name = "capa2"))
model.add(Dense(4, activation = "relu", name = "capa3"))
model.add(Dense(2, activation = "tanh", name = "capa4"))
model.add(Dense(1, activation = "softmax", name = "capaFinal"))

model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])

stop = LambdaCallback(on_epoch_end = lambda epoch, logs:
    setattr(model, 'stop_training', True) if logs['accuracy'] >= 1.0 else None)

history = model.fit(training_data, target_data, epochs = 5000, callbacks = [stop])

scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%"%(model.metrics_names[1], scores[1]*100))

print(model.predict(training_data).round())

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Accuracy del modelo')
plt.ylabel('Accuracy')
plt.xlabel('Epocas')
plt.legend(['accuracy', 'loss'], loc ='upper left')
plt.show()