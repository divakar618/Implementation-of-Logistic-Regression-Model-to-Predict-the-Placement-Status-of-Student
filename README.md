# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.


2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.


3.Import LabelEncoder and encode the dataset.


4.Import LogisticRegression from sklearn and apply the model on the dataset.


5.Predict the values of array.


6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.


7.Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DIVAKAR R
RegisterNumber: 212222240026
import pandas as pd
data = pd.read_csv('dataset/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:, :-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![Screenshot 2023-05-21 123441](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/3d87e4fd-12b3-4441-adaf-64f7a109eefe)

![Screenshot 2023-05-21 123500](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/94a0b754-c00d-4435-87df-5d233f9e93e2)
![Screenshot 2023-05-21 123519](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/8a7cc524-9b7f-4d96-a464-01d19f2b5e2a)
![Exp4_5](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/954dc236-449a-4990-8bd6-090280e2f84e)

![Exp4_6](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/b7434d57-d259-41a9-a728-a31f996aed86)

![Exp4_7](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/7e588db0-4cd7-47fd-a60e-5957519782c2)
![Exp4_8](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/97765a66-b40d-40a3-b202-da4d8b88b2a8)

![Exp4_9](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/d9db83d7-7e44-4b57-b30e-00b3d801529c)
![Exp4_10](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/88bb9f10-171a-4d6b-9553-534ad8e6855f)
![Exp4_11](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/436be248-d500-4d43-9a89-98d3b4ee1c87)

![Exp4_12](https://github.com/divakar618/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121932143/3c71bbba-6d7a-4a86-bf1c-b8b156d5e9f8)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
