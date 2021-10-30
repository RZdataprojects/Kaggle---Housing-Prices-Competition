# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data
iowa_data = pd.read_csv('train.csv')
y = iowa_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',
            'Fireplaces', 'BedroomAbvGr', 'YearRemodAdd', 'OverallCond', 'YearRemodAdd',
            'GarageArea', 'GarageCars']
X = iowa_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Regressor Model: "+ str(rf_val_mae))
