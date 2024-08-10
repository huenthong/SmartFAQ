import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Replace with your actual OpenAI API key
openai.api_key = st.secrets["mykey"]

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    question_embeddings = np.array([np.fromstring(embedding[1:-1], sep=',') for embedding in df['Question_Embedding']])
    return df, question_embeddings

df, question_embeddings = load_data()

# Check the shape and type of the embeddings
st.write(f"Shape of question_embeddings: {question_embeddings.shape}")
st.write(f"Type of question_embeddings: {type(question_embeddings)}")
st.write(f"Example embedding: {question_embeddings[0]}")

# Load the embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_most_relevant_answer(user_question, threshold=0.75):
    # Generate an embedding for the user's question
    user_question_embedding = embedding_model.encode([user_question])

    # Debug statements for user question embedding
    st.write(f"Shape of user_question_embedding: {user_question_embedding.shape}")
    st.write(f"Type of user_question_embedding: {type(user_question_embedding)}")
    st.write(f"User question embedding: {user_question_embedding}")

    # Calculate cosine similarities
    similarities = cosine_similarity(user_question_embedding, question_embeddings)

    # Debug statements for similarities
    st.write(f"Shape of similarities: {similarities.shape}")
    st.write(f"Similarities: {similarities}")

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
