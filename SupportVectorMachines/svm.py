from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler 

veriler = pd.read_csv('dataset/veriler.csv')
x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

svc = SVC(kernel='rbf')

sc= StandardScaler()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)
svc.fit(X_train,y_train);

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print('SVC')
print(cm)