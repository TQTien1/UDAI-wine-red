#Import thư viện
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score,precision_score,recall_score
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

wine = pd.read_csv('winequality-red.csv', sep = ";")

X=wine.drop('quality', axis = 1)
y=wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state= 2)

std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test =  std.transform(X_test)

os =SMOTE(random_state=0)
os_X_train,os_y_train= os.fit_resample(X_train,y_train)
os_y_train.value_counts()

new_model = KNeighborsClassifier(n_neighbors= 4)
new_model.fit(os_X_train, os_y_train)
new_pred = new_model.predict(X_test)
new_score = accuracy_score(y_test,new_pred)
print(new_score)
print(metrics.classification_report(y_test, new_pred))