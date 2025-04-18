import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# df = pd.read_csv('ratings.csv')
# df['genres'] = df['genres'].str.split('|')
# mlb = MultiLabelBinarizer()
# genre_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_)
# df = pd.concat([df, genre_encoded], axis=1)

# genre_cols = mlb.classes_.tolist()
# df['genre_string'] = df[genre_cols].astype(str).agg(''.join, axis=1)

# extracting title and year from columm
# df[['title', 'year']] = df['title'].str.extract(r'^(.*)\s\((\d{4})\)$')

# df.to_csv('ratings_updated_1.csv', encoding='utf-8')

# splitting data into training and testing based on user and timstamp

# Load and sort the dataset
df = pd.read_csv('ratings_updated_1.csv', index_col=0)

# filter out illegal characters that the model can't process
allowed_pattern = r"^[A-Za-z0-9.,!?;:'\"()\- ]+$"
df = df[df['title'].astype(str).str.match(allowed_pattern)]

# remove users whose age is less than 15 (age that is too young could be due to data input error)
df = df[df['age'] >= 15]

# replace gender strings of F and M with 0 and 1
df['gender'] = df['gender'].str.strip().str.upper()
df['gender'] = df['gender'].replace('F',0)
df['gender'] = df['gender'].replace('M',1)
df = df.sort_values(by=['user_id', 'timestamp'])

# Count the number of interactions per user
df['interaction_count'] = df.groupby('user_id')['movie_id'].transform('count')

# Rank each row per user (chronological order)
df['rank'] = df.groupby('user_id').cumcount()

# Compute training mask: keep first 80% for each user
df['is_train'] = df['rank'] < (0.8 * df['interaction_count'])

# Split without loop
train_df = df[df['is_train']].drop(columns=['interaction_count', 'rank', 'is_train']).reset_index(drop=True)
test_df = df[~df['is_train']].drop(columns=['interaction_count', 'rank', 'is_train']).reset_index(drop=True)
train_df.to_csv('training_testing/train.csv',encoding='utf-8')
test_df.to_csv('training_testing/test.csv',encoding='utf-8')
