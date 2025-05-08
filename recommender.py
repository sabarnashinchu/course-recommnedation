import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load all pre-trained components
courses_df = pd.read_csv(os.path.join(MODEL_DIR, 'courses_cleaned.csv'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, 'tfidf_matrix.pkl'))

user_map = joblib.load(os.path.join(MODEL_DIR, 'user_map.pkl'))
course_map = joblib.load(os.path.join(MODEL_DIR, 'course_map.pkl'))
reverse_course_map = joblib.load(os.path.join(MODEL_DIR, 'reverse_course_map.pkl'))
user_features = joblib.load(os.path.join(MODEL_DIR, 'user_features.pkl'))
course_features = joblib.load(os.path.join(MODEL_DIR, 'course_features.pkl'))


# Clean course name for better display
def clean_course_name(raw_name):
    parts = raw_name.strip().split()
    if 'Course' in parts:
        course_idx = parts.index('Course')
        return ' '.join(parts[:course_idx])
    return raw_name


def recommend(user_input: dict, top_n=5):
    user_id = user_input.get('user_id')
    qualification = user_input.get('qualification', '').lower()
    goals = user_input.get('goals', '').lower()
    aspirations = user_input.get('aspirations', '').lower()
    strengths = user_input.get('strengths', '').lower()
    improvement = user_input.get('improvement', '').lower()

    # Use collaborative filtering if user is known
    if user_id and user_id in user_map:
        user_idx = user_map[user_id]
        user_vec = user_features[user_idx]
        scores = course_features @ user_vec
        top_indices = np.argsort(scores)[::-1][:top_n]

        recommendations = [
            {
                'course_id': reverse_course_map[idx],
                'name': clean_course_name(
                    courses_df.set_index('course_id').loc[reverse_course_map[idx], 'name']
                ),
                'score': float(scores[idx])
            }
            for idx in top_indices
        ]
        # Group by cleaned name and keep only highest score per name
        unique_recommendations = {}
        for rec in recommendations:
            name = rec['name']
            if name not in unique_recommendations or rec['score'] > unique_recommendations[name]['score']:
                unique_recommendations[name] = rec

        final_recommendations = list(unique_recommendations.values())[:top_n]
        return final_recommendations

    # Otherwise use content-based filtering
    combined_text = ' '.join([qualification, goals, aspirations, strengths, improvement])
    if not combined_text.strip():
        return []

    user_vec = tfidf.transform([combined_text])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sim_scores)[::-1][:top_n]

    recommendations = [
        {
            'course_id': courses_df.iloc[i]['course_id'],
            'name': clean_course_name(courses_df.iloc[i]['name']),
            'score': float(sim_scores[i])
        }
        for i in top_indices if sim_scores[i] > 0
    ]

    # Group by cleaned name and keep only highest score per name
    unique_recommendations = {}
    for rec in recommendations:
        name = rec['name']
        if name not in unique_recommendations or rec['score'] > unique_recommendations[name]['score']:
            unique_recommendations[name] = rec

    final_recommendations = list(unique_recommendations.values())[:top_n]
    return final_recommendations
