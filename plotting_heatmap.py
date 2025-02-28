from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
def load_local_data(name):
    path = f'/path/Dataset/{name}.csv'
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

# Load the data
train_df = load_local_data('train')

# List of features to be used
features = [
    'square_feet', 'year_built', 'floor_count', 'meter',
    'air_temperature', 'cloud_coverage', 'dew_temperature',
    'sea_level_pressure', 'wind_direction', 'wind_speed'
]

# Check available features
available_features = [f for f in features if f in train_df.columns]

# Sample and preprocess the data
sample_data = train_df.sample(frac=0.1, random_state=42).dropna(subset=available_features)

if available_features:
    # Standardize numerical data
    scaler = StandardScaler()
    sample_data_scaled = scaler.fit_transform(sample_data[available_features])
    
    # Conduct PCA analysis
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(sample_data_scaled)
    

# Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train_df[available_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.savefig('/path/correlation_heatmap_all.svg', format='svg')
