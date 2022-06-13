# init model
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=5)


# load libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# generate random dataframe
df = pd.DataFrame({'x': np.random.randint(1, 100, 100), 'y': np.random.randint(1, 100, 100)}, columns=['x', 'y'])


# cluster
clustered = model.fit_predict(df)


# plot results
plt.scatter(df['x'], df['y'], c=clustered)
plt.savefig('agglomerative_clustering.png')
plt.show()