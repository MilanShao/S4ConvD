import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_log_error
import wandb

# Initialize WandB
wandb.init(project="name", entity="name", name="Ensemble_CPU_Benchmark")

# Load the preprocessed data
train_set = pd.read_csv('/path/Dataset/train.csv')
val_set = pd.read_csv('/path/Dataset/val.csv')
test_set = pd.read_csv('/path/Dataset/test.csv')

# Define features and target variable
features = ['hour_x', 'hour_y', 'air_temperature', 'cloud_coverage', 
            'dew_temperature', 'square_feet', 'year_built', 'site_id']
target = 'meter_reading_log1p'

X_train = train_set[features]
y_train = train_set[target]

X_val = val_set[features]
y_val = val_set[target]

X_test = test_set[features]
y_test = test_set['meter_reading']  # for RMSLE evaluation

# Train XGBoost
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Train LightGBM
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)
lgb_params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=10)
lgb_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

# Train CatBoost
cb_model = cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, loss_function='RMSE', random_seed=42, verbose=0)
cb_model.fit(X_train, y_train)
cb_preds = cb_model.predict(X_test)

# Define a simple FFNN
class FFNN(nn.Module):
    def __init__(self, input_dim):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# Prepare dataset for FFNN
ffnn_train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
ffnn_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
ffnn_train_loader = DataLoader(ffnn_train_dataset, batch_size=8, shuffle=True)

# Train FFNN
ffnn_model = FFNN(input_dim=X_train.shape[1]).to('cpu')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ffnn_model.parameters(), lr=1e-3)

# Training loop for FFNN
for epoch in range(5):
    for x_batch, y_batch in ffnn_train_loader:
        optimizer.zero_grad()
        outputs = ffnn_model(x_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

ffnn_preds = ffnn_model(ffnn_test_tensor).detach().numpy().flatten()

# Ensemble weights
weights = {'xgb': 0.3, 'lgb': 0.5, 'cb': 0.15, 'ffnn': 0.05}

# Calculate weighted average of predictions
ensemble_preds = (weights['xgb'] * np.expm1(xgb_preds) +
                  weights['lgb'] * np.expm1(lgb_preds) +
                  weights['cb'] * np.expm1(cb_preds) +
                  weights['ffnn'] * np.expm1(ffnn_preds))

# Calculate RMSLE
test_rmsle = np.sqrt(mean_squared_log_error(y_test, ensemble_preds))
print(f"Test RMSLE: {test_rmsle:.4f}")

# Log test RMSLE to WandB
wandb.log({"Test RMSLE": test_rmsle})

# Finish the wandb run
wandb.finish()