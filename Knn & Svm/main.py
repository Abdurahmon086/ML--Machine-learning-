import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def plot_decision_boundary(X, y, model, title):
    h = .02
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 4))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Datasetni yaratish
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

# Ma'lumotlarni test va train qismlarga ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN modelini yaratish va o'qitish
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# SVM modelini yaratish va o'qitish
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Decision boundary ni chizish
plot_decision_boundary(X, y, knn_model, 'KNN Classification')
plot_decision_boundary(X, y, svm_model, 'SVM Classification')

# KNN va SVM natijalari uchun confusion matrix va classification report
y_knn_pred = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_knn_pred)
print(f'KNN aniqlik: {accuracy_knn:.2f}')

y_svm_pred = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_svm_pred)
print(f'SVM aniqlik: {accuracy_svm:.2f}')

cm_knn = confusion_matrix(y_test, y_knn_pred)
cm_svm = confusion_matrix(y_test, y_svm_pred)

print('KNN Confusion Matrix:')
print(cm_knn)
print('\nKNN Classification Report:')
print(classification_report(y_test, y_knn_pred))

print('\nSVM Confusion Matrix:')
print(cm_svm)
print('\nSVM Classification Report:')
print(classification_report(y_test, y_svm_pred))

