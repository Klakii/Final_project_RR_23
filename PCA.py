# Section: Data Loading and Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_pca_correlation_graph

# Load the data
raw_data = pd.read_csv("workation.csv", sep=';')

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