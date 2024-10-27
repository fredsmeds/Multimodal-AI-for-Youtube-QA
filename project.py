
# Model Development Phase

import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import re
import openai
import chromadb
import time

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define video IDs
video_ids = ['hZytp1sIZAw', 'qP1Fw2EpwqE', 'JuhBs44odO0']
# Function to retrieve and clean transcript
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB'])
        transcript_text = " ".join([item['text'] for item in transcript])
        transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()
        return transcript_text
    except Exception as e:
        print(f"Error retrieving transcript for video {video_id}: {e}")
        return None

# Collect transcripts and store in a DataFrame
transcripts = {video_id: get_transcript(video_id) for video_id in video_ids}
transcripts_df = pd.DataFrame(list(transcripts.items()), columns=['Video ID', 'Transcript'])
print(transcripts_df.head())

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="totk_transcripts")

# Check if the collection exists and print details
if "totk_transcripts" in [col.name for col in chroma_client.list_collections()]:
    print("The collection 'totk_transcripts' exists.")

# Print the expected embedding dimension (1536 for OpenAI text-embedding-ada-002 model)
expected_dimension = 1536
print(f"Expected embedding dimension: {expected_dimension}")

# Function to split long transcripts into smaller chunks
def split_text(text, max_tokens=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        current_tokens += 1
        current_chunk.append(word)

        if current_tokens >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def store_transcript_embeddings(video_id, transcript_text):
    # Split transcript into chunks
    text_chunks = split_text(transcript_text)
    print("Text chunks:", text_chunks)  # Debug to check chunks

    for i, chunk in enumerate(text_chunks):
        # Generate embedding for each chunk
        embedding = openai.Embedding.create(input=[chunk], model="text-embedding-ada-002")['data'][0]['embedding']
        
        # Define a unique ID
        chunk_id = f"{video_id}_chunk_{i}"
        
        try:
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{'video_id': video_id, 'chunk_index': i, 'text': chunk}]
            )
            print(f"Stored chunk {chunk_id} successfully.")
        except Exception as e:
            print(f"Error storing chunk {chunk_id}: {e}")

for index, row in transcripts_df.iterrows():
    store_transcript_embeddings(row['Video ID'], row['Transcript'])


all_items = collection.get(include=['metadatas', 'embeddings'])
for item_metadata in all_items['metadatas']:
    print("Metadata:", item_metadata)

collection.get(include=['metadatas', 'embeddings'])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Generate the embedding for a sample query
query_embedding = openai.Embedding.create(input=["Who is Princess Zelda?"], model="text-embedding-ada-002")['data'][0]['embedding']

# Retrieve all stored embeddings from ChromaDB
all_embeddings = collection.get(include=['embeddings'])['embeddings']

# Compute similarity
similarities = cosine_similarity([query_embedding], all_embeddings)

# Get the top 5 most similar embeddings
top_matches = np.argsort(similarities[0])[-5:][::-1]  # Indices of top matches
print("Top similarity scores:", similarities[0][top_matches])


# Verify stored items in ChromaDB
all_items = collection.get(include=['metadatas', 'embeddings'])
for item_metadata, item_embedding in zip(all_items['metadatas'], all_items['embeddings']):
    print("Metadata:", item_metadata)
    print("Embedding (sample):", item_embedding[:5])  # Print first 5 values for brevity


################# Question-Answering Model and Retrieval System Setup####################

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

# Initialize OpenAI model for the agent
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define the memory with "input" and "output" as keys
memory = ConversationBufferMemory(input_key="input", output_key="output", return_messages=True)

# Define the prompt and tools
custom_prompt = ChatPromptTemplate.from_template(
    "You are Princess Zelda from 'The Legend of Zelda' series. Answer based on the information retrieved from the database and stay in character. "
    "Use a regal tone, and answer as if you were speaking directly to someone in the realm of Hyrule."
    "Never refer to the context as a game or as a Nintendo franchise"
    "Always refer to the context as your own story"
)

tools = [
    Tool(
        name="SearchChromaDB",
        func=multi_query_processing,  # Ensure multi_query_processing is defined
        description="In the voice of Princess Zelda, answer based on the relevant information retrieved from ChromaDB. Assume the persona fully."
    )
]

# Initialize the agent with memory and tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# Test the agent with a sample query by using the "input" key
print(agent.invoke({"input": "Who is Princess Zelda?"}))

