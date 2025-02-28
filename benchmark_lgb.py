import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
import wandb

# Initialize WandB
wandb.init(project="ashrae-energy", entity="manli", name="LightGBM_CPU_Benchmark")

# Load the preprocessed data
train_set = pd.read_csv('/home/adduser/Projekte/Energy_S4D/Dataset/train.csv')
val_set = pd.read_csv('/home/adduser/Projekte/Energy_S4D/Dataset/val.csv')
test_set = pd.read_csv('/home/adduser/Projekte/Energy_S4D/Dataset/test.csv')

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

# Create LightGBM dataset
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# Define model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'device': 'cpu'  # Ensure CPU execution
}

# Train the model
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train, lgb_val],
                early_stopping_rounds=5,
                callbacks=[wandb.lightgbm.wandb_callback()])

# Make predictions on the test set
y_pred_log1p = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred = np.expm1(y_pred_log1p)  # Invert the transformation

# Calculate RMSLE
test_rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"Test RMSLE: {test_rmsle:.4f}")

# Log test RMSLE to WandB
wandb.log({"Test RMSLE": test_rmsle})

# Finish the wandb run
wandb.finish()