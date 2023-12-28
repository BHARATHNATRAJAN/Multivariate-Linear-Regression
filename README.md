# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd

### Step2
read the csv file
### Step3
get the values of x and y variables
### Step4
create the linear regression model and fit
### Step5
print the predicted output.
## Program:
```
# DEVELOPED BY :BHARATH
# REFERANCE NUMBER : 23003413
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])

model = LinearRegression()
model.fit(X, y)

new_data = np.array([[4, 5]])
predicted_output = model.predict(new_data)

print(f'Predicted Output: {predicted_output}')



```
## Output:
![Screenshot 2023-12-28 193350](https://github.com/BHARATHNATRAJAN/Multivariate-Linear-Regression/assets/147473529/f019ceb7-2192-4413-9aef-aabd0646ca5f)


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
