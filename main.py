import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Wine.csv")

x = data.iloc[:,:13].values
y = data.iloc[:,13].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

#PCA öncesi
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

#PCA sonrası
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2, y_train)

y_pred = classifier.predict(x_test)

y_pred2 = classifier2.predict(x_test2)

#Actual / PCA olmadan çıkan sonuç
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Gerçek / PCAsiz:")
print(cm)


#Actual / PCA sonrası çıkan sonuç
cm2 = confusion_matrix(y_test, y_pred2)
print("Gerçek / PCA ile:")
print(cm2)

#PCA sonrası / PCA öncesi çıkan sonuç
cm3 = confusion_matrix(y_pred, y_pred2)
print("PCAsiz / PCAli:")
print(cm3)























