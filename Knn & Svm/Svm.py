import pandas as pd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def visualize_results(models, predictions):
    plt.figure(figsize=(10, 6))
    plt.bar(models, predictions, color=['blue', 'green', 'orange', 'red'])
    plt.title('Model natijalari')
    plt.xlabel('Model')
    plt.ylabel('Natija')
    plt.show()

# Train ma'lumotlarini yuklab olish
data_set = pd.read_csv('train.csv')

# X va Y ni ajratib olish
X = data_set.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values
Y = data_set.iloc[:,-1].values

# Decision Tree ni ishlatish
dst = tree.DecisionTreeClassifier()
dst.fit(X,Y)

# Yangi ma'lumotlarni taxmin qilish
new = [(20,295,1752,3893,10,0,7,1,1,0)]
showfirst  = (dst.predict(new))
print("Decision tree ->",showfirst)

# Test ma'lumotlarini yuklab olish
dataset = pd.read_csv("train.csv")
dataset.head()

# Test ma'lumotlari va natijalarni sinovlash
X = dataset.iloc[:,[1,3,4,7,5,8]]
X.head()
score = StandardScaler()
X = score.fit_transform(X)
X

Y = dataset.iloc[:,-1]
Y.head()

# KNN modelini o'rganish va sinovlash
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X,Y)

# Test ma'lumotlarini yuklab olish
X_test = pd.read_csv("test.csv")
X_test.head()

X_test = X_test.iloc[:,[1,3,4,7,5,8]]
x_test = score.fit_transform(X_test)
x_test

# KNN natijalarini aniqlash va ekranga chiqarish
model_knn_predict = model_knn.predict(x_test)
print("KNN ->", model_knn_predict)

# SVM modelini yaratish va o'rganish
model_svm = SVC()
model_svm.fit(X, Y)
# Test ma'lumotlariga SVM ni qo'llash
model_svm_predict = model_svm.predict(x_test)
print("SVM ->", model_svm_predict)

# Random Forest modelini o'rganish va sinovlash
model_rdf = RandomForestClassifier()
model_rdf.fit(X,Y)
model_rdf_predict = model_rdf.predict(x_test)
print("Random Forest ->", model_rdf_predict)

# Natijalarni visualizatsiya qilish
models = ['Decision Tree', 'KNN', 'SVM', 'Random Forest']
predictions = [showfirst[0], model_knn_predict[0], model_svm_predict[0], model_rdf_predict[0]]
visualize_results(models, predictions)

