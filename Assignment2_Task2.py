import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral, UserMessage
import time
from mistralai.models.sdkerror import SDKError

# Set your API key
os.environ["MISTRAL_API_KEY"] = "NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB"
api_key = os.getenv("MISTRAL_API_KEY")

# Debug: Print API key to verify it's set (remove before production)
print("MISTRAL_API_KEY:", api_key)

# Define the UDST policies (replace URLs with actual ones)
policies = {
    "ACADEMIC ANNUAL LEAVE POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
    "INTELLECTUAL PROPERTY POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
    "CREDIT HOUR POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "ACADEMIC QUALIFICATIONS POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
    "ACADEMIC PROFESSIONAL DEVELOPMENT POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-professional-development",
    "ACADEMIC MEMBERS’ RETENTION POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members’-retention-policy",
    "ACADEMIC FREEDOM POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
    "ACADEMIC CREDENTIALS POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
    "Academic Appraisal Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-procedure",
    "ACADEMIC APPRAISAL POLICY": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
}

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def load_policy_text(url):
    """Load and return text content from a given policy URL."""
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tag = soup.find("div")
    if tag:
        return tag.get_text()
    return ""

def chunk_text(text, chunk_size=512):
    """Split text into chunks of a fixed size."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_text_embedding(list_txt_chunks, retries=3, delay=2):
    client = Mistral(api_key=api_key)
    for attempt in range(retries):
        try:
            embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
            return embeddings_batch_response.data
        except SDKError as e:
            if "429" in str(e):
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e
    raise Exception("Exceeded maximum retries for embedding API call.")

@st.cache_data(show_spinner=False)
def build_index(chunks):
    """Build a FAISS index from text chunks."""
    text_embeddings = get_text_embedding(chunks)
    embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])
    d = len(text_embeddings[0].embedding)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, chunks

def embed_question(question):
    """Embed the user question."""
    question_embeddings = np.array([get_text_embedding([question])[0].embedding])
    return question_embeddings

def retrieve_chunks(question_embeddings, index, chunks, k=2):
    """Retrieve the top k chunks that match the question."""
    D, I = index.search(question_embeddings, k=k)
    retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
    return retrieved_chunks

def generate_answer(prompt, retries=3, delay=2):
    """Call the Mistral chat API to generate an answer from the prompt with retries on rate limits."""
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=prompt)]
    for attempt in range(retries):
        try:
            chat_response = client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
            )
            return chat_response.choices[0].message.content
        except SDKError as e:
            if "429" in str(e):
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e
    raise Exception("Exceeded maximum retries for chat API call.")

# --- Streamlit Interface ---

st.title("UDST Policies Chatbot")

# Policy selection
selected_policy = st.selectbox("Select a Policy", list(policies.keys()))

# Input for the user's query
user_query = st.text_input("Enter your question about the policy:")

if st.button("Get Answer"):
    if not user_query:
        st.error("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            # Load and process the selected policy text
            url = policies[selected_policy]
            policy_text = load_policy_text(url)
            if not policy_text:
                st.error("Unable to load the selected policy. Please try another policy.")
            else:
                chunks = chunk_text(policy_text, chunk_size=512)
                index, chunks = build_index(chunks)
                question_embeddings = embed_question(user_query)
                retrieved_chunks = retrieve_chunks(question_embeddings, index, chunks, k=2)
                # Create the prompt using retrieved chunks
                prompt = f"""
Below is an excerpt from the {selected_policy}:
---------------------
{retrieved_chunks}
---------------------
Using only the information provided above, answer the following question.
Question: {user_query}
Answer:
"""
                answer = generate_answer(prompt)
                st.text_area("Answer:", answer, height=300)
