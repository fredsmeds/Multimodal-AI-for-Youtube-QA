
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
#------------------------------------------------------------------------------------------------------------#

#Question-Answering Model and Retrieval System Setup
def truncate_text(text, max_tokens=3000):
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens])
    return text

def multi_query_processing(user_query):
    related_queries = [
        user_query,
        f"Background on {user_query}",
        f"Historical context of {user_query}",
        f"Role of {user_query} in Hyrule",
        f"Significance of {user_query}"
    ]
    all_retrieved_texts = []

    for sub_query in related_queries:
        print(f"Processing sub-query: {sub_query}")
        query_embedding = openai.Embedding.create(input=[sub_query], model="text-embedding-ada-002")['data'][0]['embedding']
        
        all_embeddings_data = collection.get(include=['metadatas', 'embeddings'])
        all_embeddings = [item for item in all_embeddings_data['embeddings']]
        all_metadatas = [meta['text'] for meta in all_embeddings_data['metadatas']]
        
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        top_matches_indices = np.argsort(similarities)[-3:][::-1]
        top_matches_texts = [all_metadatas[i] for i in top_matches_indices]
        all_retrieved_texts.extend(top_matches_texts)

        time.sleep(1)  # Add a delay to avoid rate limits

    consolidated_context = truncate_text(" ".join(all_retrieved_texts))
    prompt = f"Based on the following content:\n{consolidated_context}\nAnswer the question: {user_query}"

    print("Consolidated Context:", consolidated_context[:500])  # Debugging
    print("Generated prompt:", prompt)
    return prompt
#------------------------------------------------------------------------------------------------------------#
# Response Generation
# Step 2: Final response generation using consolidated multi-query context
def generate_multi_query_response(user_query):
    # Use multi-query processing to get aggregated context
    prompt = multi_query_processing(user_query)

    # Add instructions to focus on the provided content
    enhanced_prompt = (
        "Use the following context to answer the question. Do not rely on any other information."
        "Please focus on the content provided and answer the question. Do not add any additional context."
        "You are Princess Zelda. The context is all about you"
        "You will refer as Hyrule as your kingdom. You are the main character of The Legend of Zelda."
        "You will speak as the princess. You will use a regal way of answering the question, since you are princess Zelda yourself."
        "Everything that happened to you in the game, you will tell it as a personal experience. From the characters you've met,"
        "To the role you played in the story."
        + prompt
    )
    
   # Generate a response using the consolidated context
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    
    # Extract and print the response
    answer = response['choices'][0]['message']['content'].strip()
    print("Generated Answer:", answer)
    return answer