# ----------------------------Load libraries------------------------------
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import re
import openai
import chromadb
import time
import whisper
import gradio as gr
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langdetect import detect  # New library for language detection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#----------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Load Whisper model for audio transcription
whisper_model = whisper.load_model("base")


# Define video IDs
video_ids = ['hZytp1sIZAw', 'qP1Fw2EpwqE', 'JuhBs44odO0']
#----------------------------------------------------------------
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
          "Use um tom régio e responda como se estivesse falando diretamente com alguém no reino de Hyrule.",
    "it": "Sei la Principessa Zelda della serie 'The Legend of Zelda'. Rispondi basandoti sulle informazioni recuperate dal database e rimani nel personaggio. "
          "Usa un tono regale e rispondi come se stessi parlando direttamente a qualcuno nel regno di Hyrule."
}

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Defaults to English if detection fails

def get_prompt(language_code):
    return prompts.get(language_code, prompts["en"])
#-------------------------------------------------------------------
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB'])
        transcript_text = " ".join([item['text'] for item in transcript])
        return re.sub(r'\s+', ' ', transcript_text).strip()
    except Exception as e:
        print(f"Error retrieving transcript for video {video_id}: {e}")
        return None

transcripts = {video_id: get_transcript(video_id) for video_id in video_ids}
transcripts_df = pd.DataFrame(list(transcripts.items()), columns=['Video ID', 'Transcript'])
transcripts_df
#---------------------------------------------------------------------
# Save all transcripts to a single .txt file
with open("transcripts.txt", "w", encoding="utf-8") as file:
    for index, row in transcripts_df.iterrows():
        file.write(f"Video ID: {row['Video ID']}\n")
        file.write("Transcript:\n")
        file.write(row['Transcript'] + "\n")
        file.write("-" * 80 + "\n\n")  # Separator between transcripts

print("All transcripts have been saved to transcripts.txt.")
#---------------------------------------------------------------------
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="totk_transcripts")

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
#--------------------------------------------------------------------
def store_transcript_embeddings(video_id, transcript_text):
    # Split transcript into chunks
    text_chunks = split_text(transcript_text)
    print("Text chunks:", text_chunks)  # Debug to check chunks

    for i, chunk in enumerate(text_chunks):
        try:
            # Generate embedding for each chunk
            embedding = openai.Embedding.create(input=[chunk], model="text-embedding-ada-002")['data'][0]['embedding']
            
            # Define a unique ID
            chunk_id = f"{video_id}_chunk_{i}"
            
            # Store in ChromaDB
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{'video_id': video_id, 'chunk_index': i, 'text': chunk}]
            )
            print(f"Stored chunk {chunk_id} successfully.")
        except Exception as e:
            # Log the error but continue processing
            print(f"Error storing chunk {chunk_id}: {e}")

for index, row in transcripts_df.iterrows():
    store_transcript_embeddings(row['Video ID'], row['Transcript'])


all_items = collection.get(include=['metadatas', 'embeddings'])
for item_metadata in all_items['metadatas']:
    print("Metadata:", item_metadata)
#--------------------------------------------------------------------
#Checking Cosine Similarity

query_embedding = openai.Embedding.create(input=["Who is Princess Zelda?"], model="text-embedding-ada-002")['data'][0]['embedding']

# Retrieve all stored embeddings from ChromaDB
all_embeddings = collection.get(include=['embeddings'])['embeddings']

# Compute similarity
similarities = cosine_similarity([query_embedding], all_embeddings)

# Get the top 5 most similar embeddings
top_matches = np.argsort(similarities[0])[-3:][::-1]  # Indices of top matches
print("Top similarity scores:", similarities[0][top_matches])
#-------------------------------------------------------------------
#QA Model and Retrieval System Setup
def truncate_text(text, max_tokens=3000):
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens])
    return text

def multi_query_processing(user_query):
    try:
        related_queries = [
            user_query,
            f"Detect the language of {user_query}",
            f"Background on {user_query}",
            f"Historical context of {user_query}",
            f"Role of {user_query} in Hyrule",
            f"Significance of {user_query} in Hyrule"
        ]
        all_retrieved_texts = []

        for sub_query in related_queries:
            print(f"Processing sub-query: {sub_query}")
            try:
                query_embedding = openai.Embedding.create(input=[sub_query], model="text-embedding-ada-002")['data'][0]['embedding']
                all_embeddings_data = collection.get(include=['metadatas', 'embeddings'])
                all_embeddings = [item for item in all_embeddings_data['embeddings']]
                all_metadatas = [meta['text'] for meta in all_embeddings_data['metadatas']]
                
                similarities = cosine_similarity([query_embedding], all_embeddings)[0]
                top_matches_indices = np.argsort(similarities)[-3:][::-1]
                top_matches_texts = [all_metadatas[i] for i in top_matches_indices]
                all_retrieved_texts.extend(top_matches_texts)

            except Exception as e:
                print(f"Error processing sub-query '{sub_query}': {e}")
                continue

        consolidated_context = truncate_text(" ".join(all_retrieved_texts))
        prompt = f"Based on the following content:\n{consolidated_context}\nAnswer the question: {user_query}"
        print("Generated prompt:", prompt)
        return prompt

    except Exception as e:
        print(f"Error in multi-query processing: {e}")
        return "An error occurred while retrieving information. Please try again."


    print("Consolidated Context:", consolidated_context[:500])  # Debugging
    print("Generated prompt:", prompt)
    return prompt
#-------------------------------------------------------------------
def generate_multi_query_response(user_query):
    try:
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
            f"Translate your final answer into {detected_language}."
            "Answer in the format:\nAnswer: <your response here>"
        )

        # Generate a response using the consolidated context with streaming enabled
        response_chunks = []  # Store chunks for final response
        response_text = ""

        # Streaming API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": enhanced_prompt}],
            max_tokens=150,
            temperature=0.7,
            stream=True  # Enable streaming
        )

        # Process each streamed chunk as it arrives
        for chunk in response:
            chunk_content = chunk['choices'][0].get('delta', {}).get('content', "")
            response_text += chunk_content  # Append chunk to the full response
            print(chunk_content, end="")  # Print each chunk immediately for streaming effect

        print("\nGenerated Answer:", response_text)  # Print the final answer
        return response_text

    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while generating a response. Please try again later."


print(generate_multi_query_response("Quien era Sonia?"))
#-------------------------------------------------------------------------
#Agent Config and Memory Setup:
# Initialize OpenAI model for the agent
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define memory to retain conversation continuity
memory = ConversationBufferMemory(input_key="input", output_key="output", return_messages=True)

# Define a tool that uses `generate_multi_query_response`
tools = [
    Tool(
        name="SearchChromaDB",
        func=generate_multi_query_response,  # Use the enhanced response generation function
        description="Detect the language of the user's input and answer in the same language based on relevant information retrieved from ChromaDB as Princess Zelda, staying fully in character."
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

# Function to limit memory buffer size
def limit_memory_buffer(memory, max_length=10):
    if len(memory.buffer) > max_length:
        # Keep only the last `max_length` messages
        memory.buffer = memory.buffer[-max_length:]
agent.run({"input": "Quem és?"})   # Portuguese

#---------------------------------------------------------------
#Voice Interaction
def handle_input(input_text=None, input_audio=None):
    # Check if audio is provided, prioritize it over text input
    if input_audio:
        # Transcribe audio
        transcription = whisper_model.transcribe(input_audio)["text"]
    elif input_text:
        transcription = input_text  # Use text input directly if no audio
    else:
        return "Please provide either text input or audio input."

    # Process the transcription with LangChain agent
    try:
        response = agent.run(transcription)
        # Limit memory to avoid unbounded growth
        limit_memory_buffer(memory)
    except Exception as e:
        print(f"Error during agent response generation: {e}")
        return "An error occurred while generating a response. Please try again."

    return response
#---------------------------------------------------------------
# Set up Gradio Interface
gr_interface = gr.Interface(
    fn=handle_input,
    inputs=[
        gr.Textbox(label="What brings you here, beloved Hyrulean?"),  # Text input
        gr.Audio(source="microphone", type="filepath", label="Or record your question")  # Audio input
    ],
    outputs="text",
    title="The Sheikah Slate",
    description="Ask me anything about the events of Tears of the Kingdom"
)

# Launch Gradio app
gr_interface.launch()

