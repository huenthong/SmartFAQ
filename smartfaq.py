import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

openai.api_key =  st.secrets["mykey"]

# Load the dataset
df = pd.read_csv('qa_dataset_with_embeddings.csv')
question_embeddings = np.array([eval(embedding) for embedding in df['Question_Embedding']])

# Load the embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_most_relevant_answer(user_question, threshold=0.75):
    # Generate an embedding for the user's question
    user_question_embedding = embedding_model.encode([user_question])

    # Calculate cosine similarities
    similarities = cosine_similarity(user_question_embedding, question_embeddings)
    
    # Find the most similar question
    max_similarity_idx = np.argmax(similarities)
    max_similarity_score = similarities[0, max_similarity_idx]

    if max_similarity_score >= threshold:
        answer = df.iloc[max_similarity_idx]['Answer']
        return answer, max_similarity_score
    else:
        return None, None

# Streamlit interface
st.title("Health QA System")

# Text input for user question
user_question = st.text_input("Ask a question about heart, lung, and blood-related health topics:")

if st.button("Get Answer"):
    if user_question:
        answer, similarity_score = get_most_relevant_answer(user_question)
        if answer:
            st.write(f"**Answer:** {answer}")
            st.write(f"**Similarity Score:** {similarity_score:.2f}")
        else:
            st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
    else:
        st.write("Please enter a question.")

if st.button("Clear"):
    st.experimental_rerun()

# Optional additional features
st.subheader("Common FAQs")
st.write("Here are some common questions you might find useful:")
common_faqs = df['Question'].sample(5).values
for faq in common_faqs:
    st.write(f"- {faq}")

st.subheader("Rate the answer")
rating = st.slider("Was the answer helpful?", 1, 5, 3)
st.write(f"Your rating: {rating}")

  



