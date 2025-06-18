## Implementation-of-Linear-Regression-Using-Gradient-Descent
### AIM:

To write a program to predict the profit of a city using the linear regression model with gradient descent.
### Equipments Required:

    Hardware – PCs
    Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm


Implementation-of-Linear-Regression-Using-Gradient-Descent
Give a detailed algorithm 
ChatGPT said:
Here is a detailed algorithm for the Implementation of Linear Regression using Gradient Descent:

✅ Linear Regression using Gradient Descent – Detailed Algorithm
🔷 Goal
Predict a continuous output y from input features X using a linear function
^
​
 =w⋅x+b
Where:

w
w is the weight vector

b
b is the bias

x
x is the input feature vector

We minimize the Mean Squared Error (MS
2
 
🔽 Step-by-Step Algorithm
Step 1: Initialize Parameters
Randomly initialize weights 

Set number of iterations (epochs)

w = 0 (or small random value)
b = 0
alpha = 0.01
epochs = 1000
Step 2: Prepare Data
Input: Feature matrix 
n
n: number of samples

m
m: number of features

Step 3: Loop until convergence (Gradient Descent)
Repeat for a fixed number of epochs:

Step 3.1: Compute Pred
y_pred = np.dot(X, w) + b
Step 3.2: Compute Gradients
Gradient of Loss w.r.t. weights 

dw = (2/n) * np.dot(X.T, (y_pred - y))
db = (2/n) * np.sum(y_pred - y)
Step 3.3: Update Parameters
 
w = w - alpha * dw
b = b - alpha * db
Step 3.4 (Optional): Track Loss
Compute MSE at every few iterations to monitor performance.
 
Step 4: Final Output
After convergence, return learned parameters 

Use them to make predictions:

def predict(X_new):
    return np.dot(X_new, w) + b
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
