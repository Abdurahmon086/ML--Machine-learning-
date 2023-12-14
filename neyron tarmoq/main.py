import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Iris ma'lumotlarini yuklash
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Ma'lumotlarni kategoriyal holatga o'tkazish
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

# Ma'lumotlar to'plamini ta'lim va sinov bo'limlarga bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Neyron tarmoq modelini qurish
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Modelni tuzish
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modelni ta'lim etish va ta'lim tarixini saqlash
history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_data=(X_test, y_test))

# Modelni test to'plamida baholash
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Test to'plamiga o'xshash natijalarni oluvchi qismi
y_pred = model.predict(X_test)

# Ba'zi misollarni ko'rsatish
print("Haqiqiy vs Bashorat:")
for i in range(5):
    haqiqiy_class = encoder.inverse_transform(y_test[i:i+1])[0]
    bashorat_class = encoder.inverse_transform(y_pred[i:i+1])[0]
    print(f"Namuna {i+1}: Haqiqiy={haqiqiy_class}, Bashorat={bashorat_class}")

# Ta'lim tarixini ko'rsatish
plt.figure(figsize=(12, 4))

# Zarar (loss) grafiki
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Ta\'lim Zarari')
plt.plot(history.history['val_loss'], label='Sinov Zarari')
plt.title('Ta\'lim davri boyunca Zarar')
plt.xlabel('Epoxa')
plt.ylabel('Zarar')
plt.legend()

# Aniqlik (accuracy) grafiki
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Ta\'lim Aniqligi')
plt.plot(history.history['val_accuracy'], label='Sinov Aniqligi')
plt.title('Ta\'lim davri boyunca Aniqlik')
plt.xlabel('Epoxa')
plt.ylabel('Aniqlik')
plt.legend()

plt.tight_layout()
plt.show()
