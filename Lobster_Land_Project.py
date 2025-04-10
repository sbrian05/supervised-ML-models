
"""
Lobster Land - 1980s Festival Analytics Project
AD654 Spring 2024

This script contains all analyses for the Lobster Land 1980s Festival project, including:
- Summary Statistics
- Segmentation & Targeting (Clustering)
- Conjoint Analysis
- Forecasting Sony Net Income
- Classification Model (Event Preference)
- A/B Testing (Email Photos)
- Final Conclusion

Author: Brian Serwadda
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats

print("\n--- Summary Statistics ---")
imdb = pd.read_csv('imdb_movies.csv')

imdb['year'] = pd.to_numeric(imdb['year'], errors='coerce')
imdb_1980s = imdb[(imdb['year'] >= 1980) & (imdb['year'] <= 1989)]

summary = imdb_1980s.describe()
print(summary)

rating_by_genre = imdb_1980s.groupby('genre')['avg_vote'].mean()
print(rating_by_genre)

summary.to_csv('summary_statistics.csv')
rating_by_genre.to_csv('rating_by_genre.csv')


print("\n--- Segmentation & Targeting ---")
consumers = pd.read_csv('eighties_consumers.csv')

numeric_cols = consumers.select_dtypes(include=['float64', 'int64']).dropna(axis=1)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(numeric_cols)
consumers['Cluster'] = clusters

cluster_profiles = consumers.groupby('Cluster').mean()
cluster_profiles.to_csv('cluster_profiles.csv')
print(cluster_profiles)

plt.figure(figsize=(8, 6))
sns.countplot(data=consumers, x='Cluster')
plt.title('Consumer Cluster Distribution')
plt.savefig('cluster_distribution.png')
plt.close()


print("\n--- Conjoint Analysis ---")
party = pd.read_csv('party_platters.csv')
costs = pd.read_csv('vendor_costs.csv')

components = party['bundle'].str.split(',', expand=True)
components.columns = ['Starter', 'Main', 'Salad/Soup', 'Side', 'Dessert']
party = pd.concat([party, components], axis=1)

utility = {}
for column in components.columns:
    utility[column] = party.groupby(column)['avg_rating'].mean()

for category, values in utility.items():
    print(f"\nUtility Scores for {category}:")
    print(values)

recommended = {
    'Starter': utility['Starter'].idxmax(),
    'Main': utility['Main'].idxmax(),
    'Salad/Soup': utility['Salad/Soup'].idxmax(),
    'Side': utility['Side'].idxmax(),
    'Dessert': utility['Dessert'].idxmax()
}

print("\nRecommended Bundle:")
print(recommended)


print("\n--- Forecasting Sony Net Income ---")
sony = pd.read_csv('sony_financials.csv')

X = sony['Year'].values.reshape(-1, 1)
y = sony['Net Income'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[2024]])
print(f"Predicted Sony Net Income for 2024: ${prediction[0][0]:,.2f}")


print("\n--- Classification Model ---")
visitor = pd.read_csv('event_visitor.csv')

le = LabelEncoder()
visitor_encoded = visitor.apply(lambda col: le.fit_transform(col.astype(str)))

X = visitor_encoded.drop('preference', axis=1)
y = visitor_encoded['preference']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LinearRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_class = np.where(y_pred > 0.5, 1, 0)
print(confusion_matrix(y_test, y_pred_class))
print(classification_report(y_test, y_pred_class))

print("\n--- A/B Testing (Email Photos) ---")
music_pics = pd.read_csv('music_pics.csv')

anova_result = stats.f_oneway(
    music_pics[music_pics['photo'] == 'Madonna']['click_rate'],
    music_pics[music_pics['photo'] == 'Prince']['click_rate'],
    music_pics[music_pics['photo'] == 'Bruce']['click_rate'],
    music_pics[music_pics['photo'] == 'BonJovi']['click_rate']
)

print("ANOVA Test Result:", anova_result)

best_photo = music_pics.groupby('photo')['click_rate'].mean().idxmax()
print(f"Recommended Photo for Next Email: {best_photo}")


print("\n--- Conclusion ---")
print("All analyses complete! Review outputs and visualizations for insights into Lobster Land's festival strategy.")

