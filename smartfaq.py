import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

openai.api_key =  st.secrets["mykey"]

# Load the dataset and embeddings
@st.cache_data
def load_data_and_embeddings(file_path):
    df = pd.read_csv(file_path)
    
    def parse_embedding(embedding_str):
        return np.fromstring(embedding_str.strip('[]'), sep=' ')
    
    df['Question_Embedding'] = df['Question_Embedding'].apply(parse_embedding)
    embeddings = np.stack(df['Question_Embedding'].values)
    
    return df, embeddings

# Load the pre-trained embedding model
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Find the most relevant answer
def find_answer(user_question, df, embeddings, model, threshold=0.7):
    user_embedding = model.encode(user_question)
    similarities = cosine_similarity([user_embedding], embeddings)[0]
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    if best_similarity > threshold:
        return df.iloc[best_idx]['Answer']
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"

# Streamlit app
def main():
    st.title('Health Q&A Bot')

    file_path = 'qa_dataset_with_embeddings.csv'
    df, embeddings = load_data_and_embeddings(file_path)
    model = load_model()

    st.write("Ask questions about heart, lung, and blood-related health topics:")

    user_question = st.text_input("Your question:")

    if st.button("Get Answer"):
        if user_question:
            answer = find_answer(user_question, df, embeddings, model)
            st.write(answer)
        else:
            st.write("Please enter a question.")

    if st.button("Clear"):
        st.text_input("Your question:", value="", key="clear")

if __name__ == "__main__":
    main()
