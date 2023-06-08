# Section: Data Loading and Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
raw_data = pd.read_csv("/Users/nataliamiela/Documents/master/UL/clustering project/workation.csv", sep=';')

# Preview the data
print(raw_data.head())
print(raw_data.describe())

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

# Check for outliers
data.boxplot(column=data.columns.tolist())

# Correlation matrix
corr_data = data.corr()
print(corr_data)

# Normalize the data
scaler = StandardScaler()
data_s = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
