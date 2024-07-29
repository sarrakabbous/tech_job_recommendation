import streamlit as st
import pandas as pd
import numpy as np
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# Load the dataset
df1 = pd.read_csv('new.csv', on_bad_lines='skip', engine='python')
df1 = df1.dropna(subset=['jobdescription', 'jobtitle'])  # Ensure these columns do not have NaNs

# Fill missing job descriptions
df1['jobdescription'] = df1['jobdescription'].fillna('')

# TF-IDF Vectorization
tdif = TfidfVectorizer(stop_words='english')
tdif_matrix = tdif.fit_transform(df1['jobdescription'])

# Calculate cosine similarity
cosine_sim = sigmoid_kernel(tdif_matrix, tdif_matrix)

# Create an index mapping
indices = pd.Series(df1.index, index=df1['jobtitle']).drop_duplicates()

# Define recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:16]
    tech_indices = [i[0] for i in sim_scores]
    return df1['jobtitle'].iloc[tech_indices]

# Streamlit interface
st.header('Tech Jobs Recommender')

# Load pickled data
movies = pickle.load(open('job_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Dropdown list for job titles
toon_list = movies['jobtitle'].values
selected_toon = st.selectbox(
    "Type or select a job from the dropdown",
    toon_list
)

# Display recommendations
if st.button('Show Recommendation'):
    recommended_toon_names = get_recommendations(selected_toon)
    if recommended_toon_names:
        for i in recommended_toon_names:
            st.subheader(i)
    else:
        st.write("No recommendations found for the selected job title.")
