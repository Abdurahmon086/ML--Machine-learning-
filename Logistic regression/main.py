import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

# CSV faylini o'qish
data = pd.read_csv('data.csv')

# Datasetni xususiyatlari va ma'lumotlarni ajratish
X, y = data.drop('sinf', axis=1), data['sinf']

# Ma'lumotlarni test va train qismlarga ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,
random_state=42)

# Ma'lumotlarni standartlashtirish
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistik regressiya modelini tuzish
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Natijalarni vizual va jadval ko'rish
# Test ma'lumotlari bo'yicha baholash
y_pred = model.predict(X_test_scaled)

# Datasetni vizualizatsiya qilish
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='viridis')
plt.xlabel('X o\'qi')
plt.ylabel('Y o\'qi')
plt.title('Test Dataset')
plt.show()

# Accuracy ni hisoblash
accuracy = accuracy_score(y_test, y_pred)
print(f'Aniqlik: {accuracy:.2f}')

# Confusion matrix ni vizualizatsiya qilish
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Bashorat')
plt.ylabel('Haqiqiy')
plt.title('Confusion Matrix')
plt.show()
# Classification report ni ko'rish
print('Classification Report:')
print(classification_report(y_test, y_pred))
