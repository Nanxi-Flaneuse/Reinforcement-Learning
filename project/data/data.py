import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('ratings.csv')
# df['genres'] = df['genres'].str.split('|')
# mlb = MultiLabelBinarizer()
# genre_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_)
# df = pd.concat([df, genre_encoded], axis=1)

# genre_cols = mlb.classes_.tolist()
# df['genre_string'] = df[genre_cols].astype(str).agg(''.join, axis=1)

df[['title', 'year']] = df['title'].str.extract(r'^(.*)\s\((\d{4})\)$')
# print(df.head())
df.to_csv('ratings_updated_1.csv', encoding='utf-8')