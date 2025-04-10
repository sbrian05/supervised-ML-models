
"""
Car Price and Depreciation Analysis Script
Web Scraping and Analysis

This script scrapes used car listings from Cars.com, collects data on price, mileage, and location,
and performs basic analysis, including:
- Average price per location
- Price depreciation estimation via regression

Author: Brian Serwadda
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


CAR_MAKE = "nissan"
CAR_MODEL = "rogue"
ZIP_CODES = ["02115", "10001"]  
MAX_PAGES = 3  


car_names, car_prices, car_mileages, car_ratings, car_locations = [], [], [], [], []

def extract_number(text):
    if text:
        numbers = re.findall(r'\d+', text.replace(',', ''))
        return int(numbers[0]) if numbers else None
    return None

def scrape_page(url, location):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    listings = soup.find_all('div', class_='vehicle-card')

    for listing in listings:
        name = listing.find('h2')
        price = listing.find('span', class_='primary-price')
        mileage = listing.find('div', class_='mileage')
        rating = listing.find('span', class_='sds-rating__count')

        car_names.append(name.text.strip() if name else None)
        car_prices.append(extract_number(price.text) if price else None)
        car_mileages.append(extract_number(mileage.text) if mileage else None)
        car_ratings.append(float(rating.text.strip()) if rating else None)
        car_locations.append(location)

for zip_code in ZIP_CODES:
    print(f"Scraping data for ZIP: {zip_code}")
    for page in range(1, MAX_PAGES + 1):
        url = f"https://www.cars.com/shopping/results/?list_price_max=&makes[]={CAR_MAKE}&maximum_distance=20&models[]={CAR_MODEL}&page={page}&page_size=20&stock_type=used&zip={zip_code}"
        scrape_page(url, zip_code)
        time.sleep(1)  

data = pd.DataFrame({
    'Name': car_names,
    'Price': car_prices,
    'Mileage': car_mileages,
    'Rating': car_ratings,
    'Location': car_locations
})

data.dropna(subset=['Price', 'Mileage'], inplace=True)

data.to_csv('car_data.csv', index=False)

print("\nCollected Data Preview:")
print(data.head())

avg_price = data.groupby('Location')['Price'].mean()
print("\nAverage Price by Location:")
print(avg_price)

plt.figure(figsize=(8, 6))
for location in data['Location'].unique():
    subset = data[data['Location'] == location]
    plt.scatter(subset['Mileage'], subset['Price'], label=f'ZIP {location}', alpha=0.6)

plt.title('Price vs Mileage by Location')
plt.xlabel('Mileage')
plt.ylabel('Price ($)')
plt.legend()
plt.savefig('price_vs_mileage.png')
plt.close()

X = data['Mileage'].values.reshape(-1, 1)
y = data['Price'].values.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X, y)

print("\nRegression Coefficient (Depreciation per mile):", reg.coef_[0][0])
print("Intercept (Base Price):", reg.intercept_[0])

mileage_3_year = np.array([[36000]])
predicted_price = reg.predict(mileage_3_year)
print(f"\nPredicted price for ~36,000 miles: ${predicted_price[0][0]:,.2f}")


print("\nScript completed successfully. Check 'car_data.csv' and 'price_vs_mileage.png'.")
