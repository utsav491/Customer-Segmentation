# K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans 
# Importing the dataset
dataset = pd.read_csv('C:\\Users\\utsav\\Desktop\\Mall_Customers.csv')
X1 = dataset.iloc[:, [3, 4]].values

#Plot styling
import seaborn as sns; sns.set()  # for plot styling
%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#Explore the dataset
print(dataset.head())#top 5 columns
len(dataset) # of rows

#Visualizing the data - displot
plot_income = sns.distplot(dataset["Annual Income (k$)"])
plot_spend = sns.distplot(dataset["Spending Score (1-100)"])
plt.xlabel('Income / spend')
plt.show()

#Violin plot of Income and Spend
f, axes = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)

v1 = sns.violinplot(data=dataset, x='Annual Income (k$)', color="skyblue",ax=axes[0])
v2 = sns.violinplot(data=dataset, x='Spending Score (1-100)',color="lightgreen", ax=axes[1])
v1.set(xlim=(0,420))
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X1)
print(y_kmeans)
#plt.scatter()


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\utsav\\Desktop\\Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
print(dataset.columns)
plt.scatter(dataset["Annual Income (k$)"] , dataset["Spending Score (1-100)"], color  = 'black')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.show()


# Visualising the clusters
plt.scatter(X1[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X1[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X1[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X1[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.show()


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
