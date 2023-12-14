import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Datasetni shakllantirish
X, _ = make_blobs(n_samples=20, n_features=10, centers=3, random_state=42)

# K-means klasterlash modeli orqali datasetni klasterlash algoritmini va dasturini ishlab chiqish
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Natijalarni baholash va ko'rish
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Natijalarni vizual ko'rsatish
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Klaster markazlari')
plt.title('K-means Klasterlash Natijalari')
plt.legend()
plt.show()

# Natijalarni DataFrame'ga o'tkazish
columns = [f'Feature {i+1}' for i in range(X.shape[1])]
data = pd.DataFrame(X, columns=columns)
data['Klaster'] = labels

# Klaster markazlarini DataFrame'ga qo'shish
centers_df = pd.DataFrame(centers, columns=columns)
centers_df['Klaster'] = ['Markaz ' + str(i+1) for i in range(len(centers))]

# Natijalarni va klaster markazlarini jadval ko'rsatish
print("Natijalar:\n", data.head())
print("\nKlaster Markazlari:\n", centers_df)


