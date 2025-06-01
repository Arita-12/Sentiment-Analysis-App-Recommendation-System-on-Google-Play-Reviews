
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
from surprise.accuracy import rmse
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load data
df = pd.read_csv("apps_reviews.csv")

# ------------- PART 1: SENTIMENT CLASSIFIER -------------

# Create sentiment labels (1 for positive, 0 for negative)
df['sentiment'] = df['review_score'].apply(lambda x: 1 if x >= 4 else 0)

# Clean the text
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    return " ".join([word for word in text.split() if word not in stop_words])

df['cleaned_review'] = df['review_text'].apply(clean_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Sentiment Classifier Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ------------- PART 2: RECOMMENDER SYSTEM -------------

# --- Collaborative Filtering ---
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['app_id', 'review_score', 'review_score']], reader)
trainset, testset = surprise_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
print("Collaborative Filtering RMSE:")
rmse(predictions)

def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[int(uid)].append((int(iid), est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_recommendations = get_top_n(predictions, n=5)
print("Sample Collaborative Recommendations for user 1:")
print(top_recommendations.get(1, []))


# --- Content-Based Filtering ---
app_reviews = df.groupby('app_id')['cleaned_review'].apply(lambda x: " ".join(x)).reset_index()
tfidf_app = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_app.fit_transform(app_reviews['cleaned_review'])

cos_sim = cosine_similarity(tfidf_matrix)

# Function to get similar apps
def recommend_similar_apps(app_id, top_n=5):
    if app_id not in app_reviews['app_id'].values:
        return []
    idx = app_reviews[app_reviews['app_id'] == app_id].index[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    similar_indices = [i[0] for i in sim_scores]
    return app_reviews.iloc[similar_indices]['app_id'].tolist()

print("Content-Based Recommendations for app 1:")
print(recommend_similar_apps(1, top_n=5))
