# to perform linear regression on data provided -- to predict the body weight given the brain weight of dogs
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data 
dataframe=pd.pd_fwf('brain_body.txt')
x_values=dataframe[['Brain']]
y_values=dataframe[['Body']]

#train the model using linear regression
body_reg=linear_model.LinearRegression()
body_reg.fit(x_values,y_values)

# visualize the reesults of the trained model
plt.scatter(x_values,y_values)
plt.plot(x_values,body_reg.predict(x_values))
plt.show()