import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Ma'lumotlarni yuklab olish
(x_train, y_train), (_, _) = mnist.load_data()

# Ma'lumotlarni formatlash
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train)

# Modelni tuzish
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Modelni o'qitish
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=64)

# Modelni TensorFlow.js formatida saqlash
tfjs_path = 'main.js'
tf.saved_model.save(model, tfjs_path)
