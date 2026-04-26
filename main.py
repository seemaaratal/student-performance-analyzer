import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("students.csv")

# Select marks
X = df[['Math', 'Science', 'English']]

# Apply K-Means
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X)

# Label clusters
def label_cluster(c):
    if c == 0:
        return "Top Performer"
    elif c == 1:
        return "Average"
    else:
        return "Needs Improvement"

df['Performance'] = df['Cluster'].apply(label_cluster)

print(df)

# Graph
plt.scatter(df['Math'], df['Science'], c=df['Cluster'])
plt.scatter(df['Math'], df['Science'], c=df['Cluster'])

# Names
for i, txt in enumerate(df['Name']):
    plt.annotate(txt, (df['Math'][i], df['Science'][i]))

# Centroids
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=200, c='red', marker='X')

plt.xlabel("Math")
plt.ylabel("Science")
plt.title("Student Clustering")
plt.show()
plt.xlabel("Math")
plt.ylabel("Science")
plt.title("Student Clustering")
plt.show()