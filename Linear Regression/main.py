import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def my_normalize(data):
    """Normalizatsiya funksiyasi"""
    return [(x - min(data)) / (max(data) - min(data)) for x in data]

class LinearRegression:
    """Linear Regression"""
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.weights_history = []  # O'lchovlarni saqlash
        self.bias_history = []     # Biasni saqlash
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Agar tushunchalarga o'xshagan bo'lmasa, xususiyatlarni normalizatsiya qilamiz
        X_normalized = X if np.max(X) > 1 else np.array([my_normalize(X[:,i]) for i in range(n_features)]).T

        for _ in range(self.n_iters):
            y_pred = np.dot(X_normalized, self.weights) + self.bias
            
            gradient_weights = (1/n_samples) * np.dot(X_normalized.T, (y_pred - y))
            gradient_bias = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - self.lr * gradient_weights
            self.bias = self.bias - self.lr * gradient_bias
            
            self.weights_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Ma'lumotlar to'plamini olish va o'qish va test to'plamlarga bo'lish
data = pd.read_csv("Salary_Data.csv")  # Yo'lovchi yo'lini yangilang
X = data[['YearsExperience']].values  # Agar ustun nomi 'YearsExperience' bo'lsa
y = data['Salary'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Model koefitsiyentlari va MSE ni chiqarish
print("Model Koefitsiyentalari:")
print("O'lchovlar:", model.weights)
print("Bias:", model.bias)

bashoratlar = model.predict(X_test)
mse = np.mean((y_test - bashoratlar) ** 2)
print("Mean Squared Error:", mse)

# Chizish
plt.scatter(X_train, y_train, color='red', s=20, label="O'qitish nuqtalari")
plt.scatter(X_test, y_test, color='green', s=20, label="Sinov nuqtalari")
plt.plot(X_test, bashoratlar, color='blue', linewidth=3, label='Bashorat')
plt.title("Normalizatsiya Qilingan Linear Regression")
plt.xlabel("Tajribali Yillar")
plt.ylabel("Maosh")
plt.legend()
plt.show()
