from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
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

# Sample and preprocess the data (smaller fraction)
sample_data = train_df.sample(frac=0.01, random_state=42).dropna(subset=available_features)

if available_features:
    # Standardize numerical data
    scaler = StandardScaler()
    sample_data_scaled = scaler.fit_transform(sample_data[available_features])
    
    # Conduct t-SNE analysis
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(sample_data_scaled)
    
    # t-SNE Plot with improved color palette
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=sample_data['primary_use'], palette='tab10')
    plt.title('t-SNE Plot for Building Data Classes')
    plt.savefig('/path/tsne_plot.svg', format='svg')
