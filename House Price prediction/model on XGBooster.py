import matplotlib.pyplot as plt
from xgboost import plot_importance
import  pandas as pd
from xgboost import XGBRegressor
import pickle

#Dataset
data = pd.DataFrame({
    'Area': [1500, 1800, 2400, 3000],
    'Bedrooms': [3, 4, 4, 5],
    'Bathrooms': [2, 3, 3, 4],
    'Location': [10, 25, 50, 100],
    'Price': [300000, 400000, 500000, 650000]
})

# Correlation
correlation = data[['Bedrooms', 'Area', 'Bathrooms', 'Location', 'Price']].corr()['Price']
print("Correlation between features and Price:")
print(correlation)

#feature and target variable
x = data[['Area', 'Bedrooms', 'Bathrooms', 'Location']]
y = data['Price']

#MODEL TRAINING
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(x, y)

#Save the model
pickle.dump(model, open('house_price_xgb_model.pkl', 'wb'))
print("XGBoost model saved successfully!")

#Even Better Visualization of feature importance
plot_importance(model)
plt.show()