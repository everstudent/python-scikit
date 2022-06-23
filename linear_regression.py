# load libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# train & test data
x = np.arange(1, 101)
y = 2 * x - 7 + np.random.randint(-15,20,100)
df = pd.DataFrame({"x":x,"y":y});


# init LinearRegression model
from sklearn import linear_model
model = linear_model.LinearRegression()


# train model on first 75 data points
model.fit(df.drop('y',axis=1)[:74], df['y'][:74])

# predict test on last 25 points
predict_y = model.predict(df.drop('y',axis=1)[75:])


# plot results
plt.scatter(df['x'][:74], df['y'][:74], c='g')
plt.scatter(df['x'][75:], df['y'][75:], c='b')
plt.scatter(df['x'][75:], predict_y, c='r')
plt.savefig('linear_regression.png')
plt.show()