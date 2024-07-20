# import all the libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print('pandas version: ', pd.__version__)
print('matplotlib version: ', matplotlib.__version__)
print('seaborn version: ', sns.__version__)
print('numpy version: ', np.__version__)

data=pd.read_csv('student_scores.csv')
data.head()

from google.colab import drive
drive.mount('/content/drive')

plt.figure(figsize=(9,5))
sns.pairplot(data,x_vars=['Hours'],y_vars=['Scores'],size=7,kind='scatter')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Score Prediction')
plt.show()

df = data.copy()
print(df.shape)
duplicate_rows_before = df[df.duplicated()]
duplicate_rows_before
df = df.drop_duplicates()
print(df.shape)

df.isna().sum()

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x="Hours", data=df)

plt.xlabel("Hours of study")
plt.ylabel("Scores")
plt.title("Boxplot Score Prediction")

plt.show()

X= data['Hours']
X.head()

#scores
Y=data['Scores']
Y.head()

#import machine learning data from scikit learn
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=42)

!pip install scikit-learn

#import linier regration from scikit learn
from sklearn.linear_model import LinearRegression

X_train=np.array(X_train)[:,np.newaxis]
X_test=np.array(X_test)[:,np.newaxis]

Y_train

X_train_reshaped = X_train.reshape(18, 1)

Y_train_reshaped = Y_train

lr_model = LinearRegression()
lr_model.fit(X_train_reshaped, Y_train_reshaped)

X_test_reshaped = X_test.reshape(7, 1)

y_pred=lr_model.predict(X_test_reshaped)

c=[i for i in range (1,len(Y_test)+1,1)]
plt.plot(c,Y_test,color='r',linestyle='-',label='actual data')
plt.plot(c,y_pred,color='b',linestyle='dashed',label='Prediction')
plt.xlabel('Hours')
plt.ylabel('index')
plt.title('Prediction with Linear Regression')
plt.legend()
plt.show()

#import metriks from scikit learn
from sklearn.metrics import r2_score,mean_squared_error

rsq=r2_score(Y_test,y_pred)

print('r Squared linear regression', rsq)

print('intercept of Linear regression model:',lr_model.intercept_)
print('koefisien of the line Linear regression :',lr_model.coef_)
#y=mx+c //m=gradient , x=hours, c=intercept

"""conclusion = Y= 9.71409219 X + 2.4803670915057623

DECISION TREE MODEL
"""

from sklearn.tree import DecisionTreeRegressor

dt_model= DecisionTreeRegressor()
dt_model.fit(X_train_reshaped,Y_train)

y_pred_dt=dt_model.predict(X_test_reshaped)

c=[i for i in range (1,len(Y_test)+1,1)]
plt.plot(c,Y_test,color='r',linestyle='-',label='actual data')
plt.plot(c,y_pred_dt,color='b',linestyle='dashed',label='Prediction')
plt.xlabel('Hours')
plt.ylabel('index')
plt.title('Prediction with Decision Tree')
plt.legend()
plt.show()

rsq_dt=r2_score(Y_test,y_pred_dt)

print('r square Desicion tree result:',rsq_dt)
