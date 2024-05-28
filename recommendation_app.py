import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify

app = Flask(__name__)

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
  return df[['title', 'summary', 'course_description', 'course_url']].iloc[course_indices]

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

      recommendations[row['title']] = subject_df[['title', 'summary', 'course_description']].iloc[course_indices].to_dict('records')

  return recommendations

@app.route('/recommend', methods=['GET'])
def recommend():
  subject = request.args.get('subject')
  recommendations = get_recommendations_by_subject(subject)
  return jsonify(recommendations)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)