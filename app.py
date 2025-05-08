from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static'))

# Load models
try:
    courses_df = pd.read_csv(os.path.join(MODEL_DIR, 'courses_cleaned.csv'))
    tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, 'tfidf_matrix.pkl'))
    user_map = joblib.load(os.path.join(MODEL_DIR, 'user_map.pkl'))
    course_map = joblib.load(os.path.join(MODEL_DIR, 'course_map.pkl'))
    reverse_course_map = joblib.load(os.path.join(MODEL_DIR, 'reverse_course_map.pkl'))
    user_features = joblib.load(os.path.join(MODEL_DIR, 'user_features.pkl'))
    course_features = joblib.load(os.path.join(MODEL_DIR, 'course_features.pkl'))
    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Error loading models:", e)
    courses_df = pd.DataFrame()

@app.route('/')
def serve_index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        user_id = data.get('user_id', '').strip()
        qualification = data.get('qualification', '').lower()
        goals = data.get('goals', '').lower()
        aspirations = data.get('aspirations', '').lower()
        strengths = data.get('strengths', '').lower()
        improvement = data.get('improvement', '').lower()

        if not any([qualification, goals, aspirations, strengths, improvement]):
            return jsonify({'error': 'There seems to be a mistake in your details'}), 400

        # Collaborative filtering
        if user_id in user_map:
            user_idx = user_map[user_id]
            user_vec = user_features[user_idx]
            scores = course_features @ user_vec
            top_indices = np.argsort(scores)[::-1][:5]
            recommended_courses = [{
                'course_id': reverse_course_map[i],
                'name': courses_df.set_index('course_id').loc[reverse_course_map[i], 'name'],
                'score': float(scores[i])
            } for i in top_indices]
        else:
            # Fallback to content-based
            combined_input = ' '.join([qualification, goals, aspirations, strengths, improvement])
            user_vec = tfidf.transform([combined_input])
            similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
            top_indices = similarity_scores.argsort()[::-1][:5]
            recommended_courses = [{
                'course_id': courses_df.iloc[i]['course_id'],
                'name': courses_df.iloc[i]['name'],
                'score': float(similarity_scores[i])
            } for i in top_indices if similarity_scores[i] > 0]

        if not recommended_courses:
            return jsonify({'error': 'There seems to be a mistake in your details'}), 200

        return jsonify(recommended_courses)
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
