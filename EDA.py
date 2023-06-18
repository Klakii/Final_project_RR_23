# Section: Data Loading and Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
raw_data = pd.read_csv("workation.csv", sep=';')

# Preview the data
display(raw_data.head())
display(raw_data.describe())

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

# Check for NAs
print(data.isna().any()) # no NAs

# Create box plots for each of the variables
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
fig, axs = plt.subplots(2, 5, figsize=(20, 10))

for i, ax in enumerate(axs.flat):
    ax.boxplot(data.iloc[:, i], flierprops=red_circle)
    ax.set_title(data.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()

# Plot correlation matrix
corr_data = data.corr()

f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_data,
    cmap=sns.color_palette("vlag", 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    square=True, ax=ax)

# Normalize the data
scaler = StandardScaler()
data_s = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
