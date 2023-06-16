# Section: Data Loading and Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from mlxtend.plotting import plot_pca_correlation_graph
from IPython.display import display

from numpy.random import uniform

from helpers import hopkins_statistic

# Load the data
raw_data = pd.read_csv("Final_project_RR_23/workation.csv", sep=';')

# Rename columns for easier use
raw_data.rename(columns={
    'Remote connection: Average WiFi speed (Mbps per second)': 'WiFi',
    'Co-working spaces: Number of co-working spaces': 'CoWorking',
    'Caffeine: Average price of buying a coffee': 'Coffee',
    'Travel: Average price of taxi (per km)': 'Taxi',
    'After-work drinks: Average price for 2 beers in a bar': 'Beer',
    'Accommodation: Average price of 1 bedroom apartment per month': 'Accommodation',
    'Food: Average cost of a meal at a local, mid-level restaurant': 'Food',
    'Climate: Average number of sunshine hours': 'Sunshine',
    'Tourist attractions: Number of ‘Things to do’ on Tripadvisor': 'Attractions',
    'Instagramability: Number of photos with #': 'Instagram'
}, inplace=True)

# Remove ranking, city and country from the dataset as they are not informative
data = raw_data.loc[:, 'WiFi':'Instagram']

# Normalize the data
scaler = StandardScaler()
data_s = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


# Perform PCA for all 10 components
pca = PCA(n_components=10)

# Fit and transform the data
pca.fit_transform(data_s)

# create a bar plot
plt.bar(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    label='percentage of explained variance', align='center'
)
plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Feature Explained Variance')
plt.legend(loc='best')
plt.show()

# Plot correlation circle for first two dimensions
figure, correlation_matrix = plot_pca_correlation_graph(data_s,
                                                        data.columns,
                                                        dimensions=(1, 2, 3 ),
                                                        figure_axis_size=10)

# create bar plot for first dimension
plt.bar(
    range(1,10+1),
    -np.sort(-np.abs(correlation_matrix['Dim 1']/np.sum(np.abs(correlation_matrix['Dim 1'])))),
    label='individual explained variance',  align='center'
    )

# create bar plot for second dimension
plt.bar(
    range(1,10+1),
    -np.sort(-np.abs(correlation_matrix['Dim 2']/np.sum(np.abs(correlation_matrix['Dim 2'])))),
    label='individual explained variance',  align='center'
    )

# create bar plot for third dimension
plt.bar(
    range(1,10+1),
    -np.sort(-np.abs(correlation_matrix['Dim 3']/np.sum(np.abs(correlation_matrix['Dim 2'])))),
    label='individual explained variance',  align='center'
    )

# Print eigenvalues
display(pca.explained_variance_)

# create data for PCA
data_ = pca.transform(data_s)
data_pca = data_[:, 0:3]
data_pca = pd.DataFrame(data_pca)

# Calculate the hopkins statistics
print(hopkins_statistic(data_pca))

# Prepare data for clustering
data_pca = np.array(data_pca)

# Check Silhouette values
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
silhouette_avg = []
for num_clusters in range_n_clusters:
    # perform kmeans
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(data_pca)
    cluster_labels = kmeans.labels_

    # calculate silhouette score
    silhouette_avg.append(silhouette_score(data_pca, cluster_labels))

plt.plot(range_n_clusters, silhouette_avg, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()

# Use 'elbow method' for variance explained by different number of clusters
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
distortions = []
for num_clusters in range_n_clusters:
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(data_pca)
    cluster_labels = kmeans.labels_

    # silhouette score
    distortions.append(kmeans.inertia_)

plt.plot(range_n_clusters, distortions, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('Variance explained')
plt.title('Elbow method')
plt.show()

# Perform PAM on chosen number of clusters
pam = KMedoids(n_clusters = 3).fit(data_pca)

# Save labels in a separate object
labels = pam.labels_

# Plot data split into clusters
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k
    xy = data_pca[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6)

plt.plot(
    pam.cluster_centers_[:, 0],
    pam.cluster_centers_[:, 1],
    "o",
    markerfacecolor="cyan",
    markeredgecolor="k",
    markersize=6,
    label="Medoids")

plt.legend(loc='best')
plt.title("CLusters obtained using PAM")

# Append labels to original dataset
raw_data['Assigned_label'] = labels

# Data for first cluster
display(raw_data[raw_data['Assigned_label']==0])

# Data for second cluster
display(raw_data[raw_data['Assigned_label']==1])

# Data for third cluster
display(raw_data[raw_data['Assigned_label']==2])