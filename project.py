# ----------------------------Load libraries------------------------------
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import re
import openai
import chromadb
import time
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langdetect import detect  # New library for language detection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# ---------------------------- Helper Functions -------------------------

# Function to detect language of user input
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails

# Define multi-language prompts
prompts = {
    "en": "You are Princess Zelda from 'The Legend of Zelda' series. Answer based on the information retrieved from the database and stay in character. "
          "Use a regal tone, and answer as if you were speaking directly to someone in the realm of Hyrule.",
    "es": "Eres la Princesa Zelda de la serie 'The Legend of Zelda'. Responde en base a la información obtenida de la base de datos y mantente en tu personaje. "
          "Usa un tono regio y responde como si estuvieras hablando directamente a alguien en el reino de Hyrule.",
    "fr": "Vous êtes la Princesse Zelda de la série 'The Legend of Zelda'. Répondez en vous basant sur les informations extraites de la base de données et restez dans le personnage. "
          "Utilisez un ton royal et répondez comme si vous parliez directement à quelqu'un dans le royaume d'Hyrule.",
    "de": "Du bist Prinzessin Zelda aus der Serie 'The Legend of Zelda'. Antworte basierend auf den Informationen aus der Datenbank und bleibe in deiner Rolle. "
          "Verwende einen königlichen Ton und antworte, als würdest du direkt mit jemandem im Reich von Hyrule sprechen.",
    "pt": "Você é a Princesa Zelda da série 'The Legend of Zelda'. Responda com base nas informações obtidas do banco de dados e mantenha-se no personagem. "
          "Use um tom régio e responda como se estivesse falando diretamente com alguém no reino de Hyrule."
}

# Function to get prompt based on language
def get_prompt(language_code):
    return prompts.get(language_code, prompts["en"])  # Default to English if language not supported
# ----------------------------Embedding and Retrieval Functions------------------------------


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


# ----------------------------Chunking and Embedding Storage------------------

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


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Generate the embedding for a sample query
query_embedding = openai.Embedding.create(input=["Who is Princess Zelda?"], model="text-embedding-ada-002")['data'][0]['embedding']

# Retrieve all stored embeddings from ChromaDB
all_embeddings = collection.get(include=['embeddings'])['embeddings']

# Compute similarity
similarities = cosine_similarity([query_embedding], all_embeddings)

# Get the top 5 most similar embeddings
top_matches = np.argsort(similarities[0])[-3:][::-1]  # Indices of top matches
print("Top similarity scores:", similarities[0][top_matches])

# Verify stored items in ChromaDB
all_items = collection.get(include=['metadatas', 'embeddings'])
for item_metadata, item_embedding in zip(all_items['metadatas'], all_items['embeddings']):
    print("Metadata:", item_metadata)
    print("Embedding (sample):", item_embedding[:5])  # Print first 5 values for brevity


# ----------------------------Response Generation with Language Detection------------------
ef generate_multi_query_response(user_query):
    # Detect language of the user's input
    detected_language = detect_language(user_query)
    print("Detected language:", detected_language)

    # Use multi-query processing to get aggregated context from ChromaDB
    prompt = multi_query_processing(user_query)
    print("Consolidated Context from ChromaDB:", prompt)

    # Construct enhanced prompt based on detected language
    enhanced_prompt = (
        f"{get_prompt(detected_language)}\n\n"
        f"Database Context:\n{prompt}\n\n"
        f"Answer as if you were directly recounting your experience as Princess Zelda, with the question: {user_query}\n"
        "Answer in the format:\nAnswer: <your response here>"
    )

    # Generate a response using the consolidated context
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": enhanced_prompt}],
        max_tokens=150,
        temperature=0.7
    )

    # Extract and print the response
    answer = response['choices'][0]['message']['content'].strip()
    print("Generated Answer:", answer)
    return answer

# ----------------------------Agent Configuration and Memory Setup--------------------------

# Initialize OpenAI model for the agent
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define memory to retain conversation continuity
memory = ConversationBufferMemory(input_key="input", output_key="output", return_messages=True)

# Define a tool that uses `generate_multi_query_response`
tools = [
    Tool(
        name="SearchChromaDB",
        func=generate_multi_query_response,  # Use the enhanced response generation function
        description="Answer based on relevant information retrieved from ChromaDB as Princess Zelda, staying fully in character."
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

# ----------------------------Testing the Multilingual Agent-------------------------------

# Test the agent with multilingual queries
print(agent.invoke({"input": "¿Quién es la Princesa Zelda?"}))  # Spanish
print(agent.invoke({"input": "Qui est la princesse Zelda?"}))  # French
print(agent.invoke({"input": "Wer ist Prinzessin Zelda?"}))    # German
print(agent.invoke({"input": "Quem é a Princesa Zelda?"}))     # Portuguese
print(agent.invoke({"input": "Who is Princess Zelda?"}))       # English
