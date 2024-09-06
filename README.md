## Implementation-of-Linear-Regression-Using-Gradient-Descent
### AIM:

To write a program to predict the profit of a city using the linear regression model with gradient descent.
### Equipments Required:

    Hardware – PCs
    Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm

1.Import pandas, numpy and mathplotlib.pyplot.

2.Trace the best fit line and calculate the cost function.

3.Calculate the gradient descent and plot the graph for it.

4.Predict the profit for two population sizes.
Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:Yadhav.G.P
RegisterNumber:21222323024

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
*/
```
### Output:

## DATA.HEAD()
![WhatsApp Image 2024-09-06 at 08 10 54_626dc5fc](https://github.com/user-attachments/assets/aa496290-0129-4a39-a7ad-087714c699e0)
## X VALUE:
![WhatsApp Image 2024-09-06 at 08 11 02_beafa2b2](https://github.com/user-attachments/assets/cca5c0f6-870b-4a84-893d-0a13ac010ea0)
## X1_SCALED VALUE:
![WhatsApp Image 2024-09-06 at 08 11 12_aac998f3](https://github.com/user-attachments/assets/9a82933d-96a9-448e-a2b8-1274436c2f4f)
## PREDICTED VALUES:
![WhatsApp Image 2024-09-06 at 08 11 17_fa94277c](https://github.com/user-attachments/assets/284e46be-3072-4896-a1b5-dd6d7f44d8e4)

### Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
