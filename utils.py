import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../backend/models')
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_and_train():
    print("🚀 Starting preprocessing...")
    
    courses_path = os.path.join(DATA_DIR, 'courses.csv')
    interactions_path = os.path.join(DATA_DIR, 'interactions.csv')
    
    courses = pd.read_csv(courses_path)
    interactions = pd.read_csv(interactions_path)
    
    courses.dropna(subset=['course_id', 'name', 'skills', 'difficulty', 'category'], inplace=True)
    courses['text'] = courses[['skills', 'category', 'difficulty']].fillna('').agg(' '.join, axis=1)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(courses['text'])
    
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(tfidf_matrix, os.path.join(MODEL_DIR, 'tfidf_matrix.pkl'))
    courses.to_csv(os.path.join(MODEL_DIR, 'courses_cleaned.csv'), index=False)

    interactions.rename(columns={'rating': 'user_rating'}, inplace=True)
    interactions = interactions.dropna(subset=['user_rating'])
    interactions = interactions[interactions['course_id'].isin(courses['course_id'])]

    print(f"✅ Users in interactions: {interactions['user_id'].nunique()}")
    print(f"✅ Courses in interactions: {interactions['course_id'].nunique()}")

    user_map = {uid: i for i, uid in enumerate(interactions['user_id'].unique())}
    course_map = {cid: i for i, cid in enumerate(interactions['course_id'].unique())}
    interactions['user_idx'] = interactions['user_id'].map(user_map)
    interactions['course_idx'] = interactions['course_id'].map(course_map)

    rating_matrix = csr_matrix((interactions['user_rating'],
                                (interactions['user_idx'], interactions['course_idx'])),
                                shape=(len(user_map), len(course_map)))

    print("📐 Rating matrix shape:", rating_matrix.shape)

    max_components = rating_matrix.shape[1] - 1
    if max_components < 2:
        raise ValueError("❌ Not enough distinct courses to train SVD. Need at least 2.")

    n_components = min(20, max_components)
    print("🔧 Training SVD with n_components =", n_components)
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_features = svd.fit_transform(rating_matrix)
    course_features = svd.components_.T

    joblib.dump(user_map, os.path.join(MODEL_DIR, 'user_map.pkl'))
    joblib.dump(course_map, os.path.join(MODEL_DIR, 'course_map.pkl'))
    joblib.dump({v: k for k, v in course_map.items()}, os.path.join(MODEL_DIR, 'reverse_course_map.pkl'))
    joblib.dump(user_features, os.path.join(MODEL_DIR, 'user_features.pkl'))
    joblib.dump(course_features, os.path.join(MODEL_DIR, 'course_features.pkl'))

    print("✅ Training complete and models saved.")
