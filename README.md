
# Mental Health Support Chatbot with Memory

This project is an AI-powered mental health support chatbot that uses Retrieval-Augmented Generation (RAG) with Mem0 and Pinecone to provide context-aware responses and store conversations for future reference.

## Features

✅ Provides empathetic, helpful responses for mental health support.

✅ Uses semantic search to retrieve relevant context from memory.

✅ Implements Long-Term Memory using Mem0 + Pinecone.

✅ Dynamically updates memory by adding each user interaction.

✅ Built with LLM integration (GEMINI).

✅ Streamlit-based interactive UI.

## System Flow

1.User Input →

2.Retrieve relevant context from Pinecone memory using semantic search →

3.Combine user input + retrieved context + prompt template →

4.Send to LLM for response generation →

5.Return bot response to user →

6.Store user message (and optionally response) in Pinecone memory for future use.

7.Update and Delete memory whenever needed

## Tech Stack

**Language**: Python

**UI**: Streamlit

**Memory**: Mem0 + Pinecone

**Embedding Model**: Gemini Embeddings

**LLM**: gemini-1.5-flash

**Deployment**: Streamlit Local


## Future Enhancement

- Add multi-user support with session tracking.

- Implement emotion detection for better responses.

- Add analytics dashboard for usage insights.

# Installation

## Prerequisites

Python 3.9+

GEMINI API Key

Pinecone API Key
    
## Setup

Clone the repository
```
git clone 
cd memory_bot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

GEMINI_API_KEY=your_GEMINI_api_key
PINECONE_API_KEY=your_pinecone_api_key

## Run the application

streamlit run chatbot_2.py
