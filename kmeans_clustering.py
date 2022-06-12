# init KMeans model
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=100)


# load libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# generate random dataframe
df = pd.DataFrame({'x': np.random.randint(1, 100, 25), 'y': np.random.randint(1, 100, 25)}, columns=['x', 'y'])


# scale 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=['x', 'y'])


# clusterize
clustered = model.fit_predict(df_scaled)


# plt.scatter(df['x'], df['y'])
plt.scatter(df_scaled['x'], df_scaled['y'], c=clustered)
plt.savefig('kmeans_clestering.png')
plt.show()