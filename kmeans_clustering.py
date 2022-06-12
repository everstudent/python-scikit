# init KMeans model
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)


# load libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# generate random dataframe
df = pd.DataFrame({'x': np.random.randint(1, 100, 25), 'y': np.random.randint(1, 100, 25)}, columns=['x', 'y'])


# cluster
clustered = model.fit_predict(df)


# plot results
plt.scatter(df['x'], df['y'], c=clustered)
plt.savefig('kmeans_clustering.png')
plt.show()