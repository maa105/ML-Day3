import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.tree import export_graphviz
def plot(dtree):
    export_graphviz(dtree, out_file='tree.dot',  
                filled=True, rounded=True,
                special_characters=True)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=150'])

np.random.seed(2)

data=pd.read_csv('creditcard.csv')
#print(data.head())
#print(data.isna().any())

#data.corrwith(data.Amount).plot.bar(figsize=(20,10), title='Correlation with Class', fontsize=15, rot=45, grid=True)
#plt.show()

corr=data.corr()
sn.set(style='white')
mask=np.zeros_like(corr,dtype=bool)

mask[np.triu_indices_from(mask)]=True

f, ax = plt.subplots(figsize=(18,15))
cmap = sn.diverging_palette(220, 10, as_cmap=True)

#sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidth=.5, cbar_kws={'shrink':.5})
#plt.show()

from sklearn.preprocessing import StandardScaler

data['NormalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)
data = data.drop(['Time'],axis=1)
#print(data.head())

x = data.iloc[:,data.columns!='Class']
y = data.iloc[:,data.columns=='Class']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# from sklearn.tree import DecisionTreeClassifier

# classifier = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', min_samples_leaf=1, min_samples_split=2)
# classifier.fit(X_train, Y_train)

# y_pred = classifier.predict(X_test)

# acc = accuracy_score(Y_test, y_pred)
# prec = precision_score(Y_test, y_pred)
# rec = recall_score(Y_test, y_pred)
# f1 = f1_score(Y_test, y_pred)

# results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy')
# classifier.fit(X_train, Y_train)

# y_pred = classifier.predict(X_test)

# acc = accuracy_score(Y_test, y_pred)
# prec = precision_score(Y_test, y_pred)
# rec = recall_score(Y_test, y_pred)
# f1 = f1_score(Y_test, y_pred)

# rf = pd.DataFrame([['RF (n=100)', acc, prec, rec, f1]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
# results = results.append(rf, ignore_index=True)

# print(results)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

Y_train = np_utils.to_categorical(Y_train)

classifier = Sequential()
classifier.add(Dense(10, kernel_initializer='uniform', activation='relu', input_dim=29))
classifier.add(Dense(5, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(.25))
classifier.add(Dense(15, kernel_initializer='uniform', activation='relu', input_dim=29))
classifier.add(Dense(7, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(.25))
classifier.add(Dense(2, kernel_initializer='uniform', activation='softmax'))


classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, batch_size=32, epochs=3)
classifier.save

y_pred = classifier.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(Y_test, y_pred)
prec = precision_score(Y_test, y_pred)
rec = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

results = pd.DataFrame([['NN', acc, prec, rec, f1]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
# results = results.append(rf, ignore_index=True)

print(results)
