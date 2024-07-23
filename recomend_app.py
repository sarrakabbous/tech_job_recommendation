import streamlit as st
import pandas as pd
import numpy as np
import string
import pickle
df = pd.read_csv('new.csv',error_bad_lines=False, engine ='python')
df = df.dropna()
from sklearn.feature_extraction.text import TfidfVectorizer

tdif = TfidfVectorizer(stop_words='english')

df['jobdescription'] = df['jobdescription'].fillna('')

tdif_matrix = tdif.fit_transform(df['jobdescription'])

tdif_matrix.shape

from sklearn.metrics.pairwise import sigmoid_kernel

cosine_sim = sigmoid_kernel(tdif_matrix, tdif_matrix)
indices = pd.Series(df.index, index=df['jobtitle']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
  idx = indices[title]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda X: X[1], reverse=True)
  sim_scores = sim_scores[1:16]
  tech_indices = [i[0] for i in sim_scores]
  return df['jobtitle'].iloc[tech_indices]



st.header('tech jobs recommender')
movies = pickle.load(open('job_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

toon_list = movies['jobtitle'].values
selected_toon = st.selectbox(
    "Type or select a job from the dropdown",
    toon_list
)


if st.button('Show Recommendation'):
    recommended_toon_names = get_recommendations(selected_toon)
    for i in recommended_toon_names:
        st.subheader(i)
        



