

import pandas as pd
import pickle 

df = pd.read_csv('covid19.csv', delimiter=";")
oversample = SMOTE()

#from sklearn.preprocessing import LabelEncoder

#lb_make = LabelEncoder()
#df["Case"] = lb_make.fit_transform(df["Case"])
#df[["Case"]].head(4)
#df

#df.head(4)

X= df.iloc[:, :-1].values
y= df.iloc[:, -1].values
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 6)


oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_train.shape


import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


lr= LogisticRegression()
lr.fit(X_train,y_train)

print('Precision:',precision_score(y_test, lr.predict(X_test), average='macro'))
print('Recall:',recall_score(y_test, lr.predict(X_test), average='macro'))
print('F1 score:',f1_score(y_test, lr.predict(X_test), average='macro'))

from sklearn.svm import SVC,LinearSVC
svm = SVC()
svm.fit(X_train,y_train)

print('Precision:',precision_score(y_test, svm.predict(X_test), average='macro'))
print('Recall:',recall_score(y_test, svm.predict(X_test), average='macro'))
print('F1 score:',f1_score(y_test,svm.predict(X_test), average='macro'))

from sklearn.ensemble import RandomForestClassifier
rf =RandomForestClassifier(n_estimators=30)
rf.fit(X_train,y_train)

print('Precision:',precision_score(y_test, rf.predict(X_test), average='macro'))
print('Recall:',recall_score(y_test, rf.predict(X_test), average='macro'))
print('F1 score:',f1_score(y_test,rf.predict(X_test), average='macro'))

from xgboost import XGBClassifier
model= XGBClassifier()
model.fit(X_train,y_train)

print('Precision:',precision_score(y_test, model.predict(X_test), average='macro'))
print('Recall:',recall_score(y_test, model.predict(X_test), average='macro'))
print('F1 score:',f1_score(y_test,model.predict(X_test), average='macro'))


from sklearn.neighbors import KNeighborsClassifier
vneigh = KNeighborsClassifier(n_neighbors=3)
vneigh.fit(X_train,y_train)
from sklearn.metrics import balanced_accuracy_score
print('Precision:',precision_score(y_test, neigh.predict(X_test), average='macro'))
print('Recall:',recall_score(y_test, neigh.predict(X_test), average='macro'))
print('F1 score:',f1_score(y_test,neigh.predict(X_test), average='macro'))
print('balanced accuracy score:',balanced_accuracy_score(y_test,neigh.predict(X_test)))



pickle.dump(rf, open('coronatest_model.pkl','wb'))

model =pickle.load(open('coronatest_model.pkl','rb'))
print(model.predict([[1, 0, 1, 0, 1, 1, 1, 0, 1, 1]]))






