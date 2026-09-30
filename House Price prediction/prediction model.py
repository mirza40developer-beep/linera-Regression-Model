import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.DataFrame({
    'Area': [1500, 1800, 2400, 3000],
    'Bedrooms': [3, 4, 4, 5],
    'Bathrooms': [2, 3, 3, 4],
    'Location': [10, 25, 50, 100],
    'Price': [300000, 400000, 500000, 650000]
})

# CORRELATION
correlation = data[['Bedrooms', 'Area', 'Bathrooms', 'Location', 'Price']].corr()['Price']
print("Correlation between features and Price:")
print(correlation)

# MODEL TRAINING
x = data[['Area', 'Bedrooms', 'Bathrooms', 'Location']]
y = data['Price']

model = LinearRegression()
model.fit(x, y)

pickle.dump(model, open('house_price_model.pkl', 'wb'))

print("Model saved successfully!")

#cost on per bedroom
print("Bedroom Effect on Price:")
print(model.coef_[1])
