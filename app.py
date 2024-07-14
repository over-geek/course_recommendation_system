import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# loading the dataset
df = pd.read_csv("dataset/edx_courses.csv")

# Dataset preprocessing
df = df.drop_duplicates()
text_columns = ['summary', 'instructors', 'subtitles', 'course_description', 'course_syllabus']
df[text_columns] = df[text_columns].fillna('not available')
df['text'] = df['title'] + ' ' + df['summary'] + ' ' + df['course_description'] + ' ' + df['course_url']

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
## compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation by title function
def get_recommenations_by_title(title, cosine_sim=cosine_sim):
  if title not in df['title'].values:
    return f"Course titled '{title}' not found"
  idx = df[df['title'] == title].index[0]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:4]
  course_indices = [i[0] for i in sim_scores]
  return df[['title', 'summary', 'course_description', 'course_url', 'subject']].iloc[course_indices]

# Recommendation by subject function
def get_recommendations_by_subject(subject, cosine_sim=cosine_sim):
  subject_df = df[df['subject'].str.contains(subject, case=False, na=False)]

  if subject_df.empty:
    return f"No courses found for the subject '{subject}'"
  subject_df = subject_df.reset_index(drop=True)
  tfidf = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf.fit_transform(subject_df['text'])
  cosine_sim_subject = linear_kernel(tfidf_matrix, tfidf_matrix)

  recommendations = {}
  for idx, row in subject_df.iterrows():
      sim_scores = list(enumerate(cosine_sim_subject[idx]))
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      sim_scores = sim_scores[1:4]

      course_indices = [i[0] for i in sim_scores]

      recommendations = subject_df[['title', 'summary', 'course_description', 'course_url', 'subject']].iloc[course_indices].to_dict('records')

  return recommendations

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    try:
        data = request.get_json()
        subjects = data.get('subjects')
        if not subjects or not isinstance(subjects, list):
            return jsonify({"error": "No subjects provided or subjects is not a list"}), 400

        # Extract subject names from the list of dictionaries
        subject_names = [subject['name'] for subject in subjects if 'name' in subject]
        if not subject_names:
            return jsonify({"error": "No valid subject names provided"}), 400

        # Generate recommendations for all subject names
        recommendations = {}
        for subject_name in subject_names:
            recommendations[subject_name] = get_recommendations_by_subject(subject_name)

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.route('/search', methods=['GET', 'POST'])
def search():
    try:
        data = request.get_json()
        course_title = data.get('title')
        if not course_title:
            return jsonify({"error": "No course title provided"}), 400

        recommendations = get_recommenations_by_title(course_title)
        if isinstance(recommendations, str):
            return jsonify({"error": recommendations}), 404

        return jsonify(recommendations.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)