# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas, numpy and mathplotlib.pyplot.

2.Trace the best fit line and calculate the cost function.

3.Calculate the gradient descent and plot the graph for it.

4.Predict the profit for two population sizes.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: yadhav G.P
RegisterNumber:212223230247  

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv('50_Startups.csv',header=None)
X=(data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
![Screenshot 2024-03-08 144452 new ml](https://github.com/sakthipriyadhanusu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393194/2933c518-d72c-4405-b34d-3e629b850e68)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
