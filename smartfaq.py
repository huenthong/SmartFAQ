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
    
    # Ensure user_embedding is 2D
    user_embedding = np.array(user_embedding).reshape(1, -1)
    
    # Debugging: Print shapes and types
    print(f"user_embedding type: {type(user_embedding)}")
    print(f"user_embedding shape: {user_embedding.shape}")
    print(f"user_embedding content: {user_embedding}")
    
    print(f"embeddings type: {type(embeddings)}")
    print(f"embeddings shape: {embeddings.shape}")
    print(f"embeddings content: {embeddings[:5]}")  # Print first 5 embeddings for brevity
    
    # Commented out the cosine_similarity call
    # try:
    #     similarities = cosine_similarity(user_embedding, embeddings)[0]
    # except ValueError as e:
    #     print(f"Error in cosine_similarity: {e}")
    #     return "There was an error processing your question. Please try again."
    
    # Temporary return for debugging
    return "Debugging complete."

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
  



