import numpy as np
import pandas as pd
df=pd.read_csv("/content/BAJAJFINSV.NS.csv")
df
df.describe()
df.info()
df.info()
x = df
x = x.drop(['Date','Close'], axis = 1)
x = np.array(x)
y=np.array(df.Close)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8)
from sklearn.linear_model import LinearRegression
model0 = LinearRegression()
model0.fit(x_train,y_train)
predict = np.array(model0.predict(x_test))
for i in range(predict.shape[0]):
  print("Actual price: ",y_test[i],"   Predicted price ",predict[i]," Difference =", y_test[i] - predict[i])
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,predict)
mae
