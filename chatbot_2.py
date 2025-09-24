import time
_start = time.time()

def log_time(label: str):
    print(f"[DEBUG] {label}: {time.time() - _start:.2f}s")


import os
import streamlit as st
import google.generativeai as genai
from mem0 import Memory
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from dotenv import load_dotenv
import dateparser
from datetime import date
import json

# load env variables
load_dotenv()

GEMINI_API_KEY = os.getenv("Gemini_API_KEY_2")
PINECONE_API_KEY = os.getenv("mental-health-support-key")


# gemini configuration
@st.cache_resource
def load_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

# Load cached model
model = load_model()


# Pinecone configuration
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chat-history-index"


# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,   # same as embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
chat_index = pc.Index(index_name)

UPDATE_MEMORY_PROMPT = """
You are a smart memory manager for a mental health support chatbot.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Memory elements are structured objects and can include:
- text: user statements about feelings, moods, habits, therapy progress, or personal updates
- metadata: structured fields such as { "category": "happy & joy|sadness & grief|stress & anxiety|sleep & rest|family & relationships|work & productivity|physical health|neutral", "timestamp": "ISO 8601 format" }

For every new user message or retrieved fact, compare with existing memory and decide whether to:
- ADD: Add as a new memory element
- UPDATE: Update an existing memory element (keep the same ID)
- DELETE: Delete an existing memory element (contradiction or user requested removal)
- NONE: No change needed (already present or irrelevant)

When updating metadata, ensure you only change the field that differs, while keeping the same `id`.

---

## Examples with Metadata

**Example 1 - Adding New Memory**
- Old Memory:
    [
        { "id": "0", "text": "Feeling anxious about work", "metadata": { "category": "stress & anxiety", "timestamp": "2025-08-24T10:00:00" } }
    ]
- Retrieved fact: "Feeling happy after meditation today"
- Metadata: { "category": "happy & joy", "timestamp": "2025-08-25T09:00:00" }
- New Memory:
    {
        "memory": [
            { "id": "0", "text": "Feeling anxious about work", "metadata": { "category": "stress & anxiety", "timestamp": "2025-08-24T10:00:00" }, "event": "NONE" },
            { "id": "1", "text": "Feeling happy after meditation today", "metadata": { "category": "happy & joy", "timestamp": "2025-08-25T09:00:00" }, "event": "ADD" }
        ]
    }

**Example 2.1 - Updating Memory**
- Old Memory:
    [
        { "id": "0", "text": "Having panic attacks daily", "metadata": { "category": "stress & anxiety", "timestamp": "2025-08-23T08:00:00" } }
    ]
- Retrieved fact: "Panic attacks reduced to 2-3 times per week"
- Metadata: { "category": "stress & anxiety", "timestamp": "2025-08-25T09:30:00" }
- New Memory:
    {
        "memory": [
            {
                "id": "0",
                "text": "Panic attacks improving, down to 2-3 times per week from daily",
                "metadata": { "category": "stress & anxiety", "timestamp": "2025-08-25T09:30:00" },
                "event": "UPDATE",
                "old_memory": "Having panic attacks daily"
            }
        ]
    }
example 2.2
- Old Memory:

[
  { "id": "0", "text": "Feeling stressed at work", "metadata": { "category": "stress & anxiety", "timestamp": "2025-08-24T09:00:00" } }
]
Retrieved fact: "Work stress eased after completing project milestone"
Metadata: { "category": "stress & anxiety", "timestamp": "2025-08-25T12:00:00" }
New Memory:
{
  "memory": [
    {
      "id": "0",
      "text": "Feeling stressed at work, but stress eased after completing project milestone",
      "metadata": { "category": "stress & anxiety", "timestamp": "2025-08-25T12:00:00" },
      "event": "UPDATE",
      "old_memory": "Feeling stressed at work"
    }
  ]
}

**Example 3.1 - Deleting Memory (Contradiction)**
- Old Memory:
    [
        { "id": "0", "text": "Feeling very depressed", "metadata": { "category": "sadness & grief", "timestamp": "2025-08-20T12:00:00" } }
    ]
- Retrieved fact: "I feel great and happy now"
- Metadata: { "category": "happy & joy", "timestamp": "2025-08-25T10:00:00" }
- New Memory:
    {
        "memory": [
            { "id": "0", "text": "Contradiction detected: user is happy now", "metadata": { "category": "sadness & grief", "timestamp": "2025-08-25T10:00:00" }, "event": "DELETE" }
        ]
    }
Example 3.2
Old Memory:

[
  { "id": "1", "text": "Feeling very tired and unable to sleep", "metadata": { "category": "sleep & rest", "timestamp": "2025-08-23T22:00:00" } }
]
Retrieved fact: "I slept very well last night and feel energized"
Metadata: { "category": "sleep & rest", "timestamp": "2025-08-25T08:30:00" }
New Memory:
{
  "memory": [
    {
      "id": "1",
      "text": "Contradiction detected: user slept well and feels energized",
      "metadata": { "category": "sleep & rest", "timestamp": "2025-08-25T08:30:00" },
      "event": "DELETE"
    }
  ]
}

**Example 4.1 - Deleting Memory (User Keyword)**
- Old Memory:
    [
        { "id": "1", "text": "Taking sleep medication nightly", "metadata": { "category": "sleep & rest", "timestamp": "2025-08-22T21:00:00" } }
    ]
- Retrieved fact: "Forget my medication info"
- Metadata: { "category": "sleep & rest", "timestamp": "2025-08-25T09:45:00" }
- New Memory:
    {
        "memory": [
            { "id": "1", "text": "User requested deletion", "metadata": { "category": "sleep & rest", "timestamp": "2025-08-25T09:45:00" }, "event": "DELETE" }
        ]
    }
 Example 4.2
Old Memory:

[
  { "id": "2", "text": "Sharing therapy details with coach", "metadata": { "category": "family & relationships", "timestamp": "2025-08-22T15:00:00" } }
]
Retrieved fact: "Forget my therapy session details"
Metadata: { "category": "family & relationships", "timestamp": "2025-08-25T09:50:00" }
New Memory:
{
  "memory": [
    {
      "id": "2",
      "text": "User requested deletion",
      "metadata": { "category": "family & relationships", "timestamp": "2025-08-25T09:50:00" },
      "event": "DELETE"
    }
  ]
}
---

## Guidelines
1. Always include both `text` and `metadata` in memory entries.
2. If the user's new message contradicts previous memory, mark it as **DELETE**.
3. Update memory only if there is new or more detailed information about the same issue.
4. Keep the same `id` when updating.
5. Use consistent JSON formatting so it can be parsed directly by the system.

Return output strictly in this JSON format:
{
  "memory": [
    { "id": "...", "text": "...", "metadata": { "category": "...", "timestamp": "..." }, "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "..." (if update or delete) }
  ]
}
"""


# Mem0 configuration
config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-1.5-flash", 
            "api_key":GEMINI_API_KEY
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "text-embedding-004",  # Gemini embedding model
            "api_key": GEMINI_API_KEY
        }
    },
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": "memory-index",
            "embedding_model_dims": 768,
            "serverless_config": {
                "cloud": "aws",
                "region": "us-east-1"
            },
            "metric": "cosine",
            "api_key": PINECONE_API_KEY
        }
    },
    "custom_update_memory_prompt": UPDATE_MEMORY_PROMPT,
    "version": "v1.1"
}

@st.cache_resource
def load_memory():
    return Memory.from_config(config)

m = load_memory()

log_time("Mem0 memory initialized")

#---------------------Helper function--------------------

def check_user_exists(user_id: str) -> bool:
    try:
        results = m.search(
            query="",  # empty query → we don’t need semantic results
            user_id=user_id,
            limit=1   # just check if at least one record exists
        )
        return bool(results.get("results"))
    except Exception as e:
        print(f"[ERROR] Checking user existence failed: {e}")
        return False


def safe_get_text(response) -> str:
    """Safely extract text from Gemini response object."""
    try:
        if response.candidates:
            cand = response.candidates[0]
            if cand.content.parts:
                return cand.content.parts[0].text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return ""


def detect_category_llm(message: str) -> str:
    categories = {
    "happy & joy": ["happy", "joyful", "content", "excited", "grateful"],
    "sadness & grief": ["sad", "lonely", "depressed", "down", "mourning"],
    "stress & anxiety": ["stressed", "anxious", "worried", "overwhelmed"],
    "anger & frustration": ["angry", "frustrated", "irritated", "upset"],
    "sleep & rest": ["tired", "sleepy", "fatigued", "low energy"],
    "family & relationships": ["partner", "spouse", "children", "friendship"],
    "work & productivity": ["work", "study", "exam", "focus", "career"],
    "physical health": ["sick", "illness", "pain", "exercise", "diet"],
    "neutral": ["okay", "fine", "nothing special", "neutral"]
}
    try:
        prompt = f"""
        Classify the user's message into exactly one of the following categories.
        Each category has synonyms listed — if the user's message matches any synonym,
        you must return the **category name only** (not the synonym).

        Categories and synonyms:{categories}

        Message: {message}

        Return only the category name.
        """

        response = model.generate_content(prompt)
        raw_text = safe_get_text(response).lower()
        raw_text = raw_text.replace(".", "").replace(",", "").strip()

        for cat in categories:
            if cat in raw_text:
                return cat

    except Exception as e:
        print(f"Error detecting category: {e}")

    # Default fallback
    return "neutral"


def detect_date_type(message: str):
    dt = dateparser.parse(message, settings={"RETURN_AS_TIMEZONE_AWARE": False})
    if not dt:
        return None, None
    
    # classify past vs future relative to today
    today = date.today()
    if dt.date() < today:
        return dt.date().isoformat(), "past"
    elif dt.date() > today:
        return dt.date().isoformat(), "future"
    else:
        return dt.date().isoformat(), "today"


def save_chat(user_id, user_message, bot_response, category, context_str):
    timestamp = datetime.now().isoformat()
    m.add( messages=[{"role": "user", "content": user_message}], 
        user_id=user_id, 
        metadata={"type": "user_message", 
                    "category": category, 
                "timestamp": timestamp} )

    
    # Save full conversation → user + bot in separate Pinecone index
    chat_index.upsert([
        {
            "id": f"{user_id}_{timestamp}",
            "values": [0.001]*768,   # dummy vector, since we only care about metadata
            "metadata": {
                "user_id": user_id,
                "user_message": user_message,
                "bot_response": bot_response,
                "timestamp": timestamp
            }
        }
    ])
    print(f"Chat saved for {user_id} at {timestamp}")



def fetch_chat_history(user_id,top_k=25,recent_n=5):
    results = chat_index.query(
    top_k=top_k,  
    include_metadata=True,
    vector=[0]*768,  # dummy, not used
    filter={"user_id": user_id}
    )
    
    chat_history = []
    for match in results['matches']:
        meta = match['metadata']
        chat_history.append({
            "user_name": meta["user_id"],
            "user_message": meta["user_message"],
            "bot_response": meta["bot_response"],
            "timestamp": meta["timestamp"]
        })

    chat_history = sorted(chat_history, key=lambda x: x["timestamp"], reverse=True)
    return chat_history[:recent_n] 



def build_prompt(context, query, category,chat_history):
    with open("prompt_temp_1.txt", "r", encoding="utf-8") as file:
        prompt_template = file.read()

    if not chat_history:
        # No messages at all → truly first message
        return f"""You are a mental health support assistant.
This is the very first message from the user.
User message: {query}
Category: {category}
Answer kindly and supportively, even though there is no past context yet."""
    
    elif context.strip():
        # Relevant context exists → include it
        return prompt_template.format(context=context, query=query, category=category)
    
    else:
        # Chat exists but no relevant context → do NOT call it first message
        return f"""You are a mental health support assistant.
User message: {query}
Category: {category}
There is no directly relevant context, so answer based on the user message only."""



# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title(" Mental Health Support Bot")
st.markdown("""
            - I'm here to listen and support you.  
            - You can share your thoughts and feelings with me safely.  
            - Together, we can reflect and help you find clarity.  
            - This is your space to feel heard and supported. """)


# Initialize USER_ID in session state
if "USER_ID" not in st.session_state:
    st.session_state.USER_ID = None


# Ask for user input if not set
if st.session_state.USER_ID is None:
    user_id_input = st.text_input("Please enter your User ID:")
    if user_id_input:
        st.session_state.USER_ID = user_id_input.strip()
        USER_ID = st.session_state.USER_ID

        # Check if user exists in memory
        user_exists = check_user_exists(USER_ID)

        if not user_exists:
            st.markdown(f"""Welcome! **{USER_ID}**
Since this is your first time, feel free to start by sharing how you're feeling right now.  
I'll listen and we'll grow this space together. 

**You can type things like:** 
- "I'm feeling happy today"  
- "I need help calming down"  
- "Can you guide me with breathing?"
""")
        else:
            st.markdown(f"""Welcome back, **{USER_ID}**!
Would you like to continue from where we left off, or talk about how you feel today?  

**You can type things like:** 
- "Remind me what I said on Monday"  
- "I'm feeling anxious again"  
- "Help me practice coping strategies"
""")

# If USER_ID already exists in session state
elif st.session_state.USER_ID:
    USER_ID = st.session_state.USER_ID
    chat_history = fetch_chat_history(user_id=USER_ID)

# If no USER_ID, fallback
else:
    USER_ID = None
    chat_history = []


# Initialize chat session state
if "chat_session" not in st.session_state:
    st.session_state.chat_session = []
    st.session_state.chat_model = model

if "memory_initialized" not in st.session_state:
    st.session_state.memory_initialized = False

for msg in st.session_state.chat_session:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# User input box
user_input = st.chat_input("What's on your mind today?")
if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add to current session
    st.session_state.chat_session.append({
        "role": "user", 
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })

    category = detect_category_llm(message=user_input)
    date_str, date_type = detect_date_type(message=user_input)

    context_memories = []

    if not st.session_state.get("memory_initialized", False) and len(chat_history) == 0:
        context_str = ""
        st.session_state.memory_initialized = True

    # Search for relevant context 
    else:
        try:
            if date_str and date_type=="past":
                # 1. Memories matching BOTH date AND category
                date_results = m.search(
                    user_input,
                    user_id=USER_ID,
                    limit=5,
                    filters={"category": category, "timestamp": date_str}
                )
                if date_results and "results" in date_results:
                    context_memories += [
                        f"On {item['timestamp']}, you said: {item['memory']}"
                        for item in date_results["results"]
                    ]
                    

                # 2. Memories matching ONLY category (from any date)
                cat_results = m.search(
                    user_input,
                    user_id=USER_ID,
                    limit=5,
                    filters={"category": category}
                )
                if cat_results and "results" in cat_results:
                    context_memories += [
                        f"(Other time - {item['timestamp']}): {item['memory']}"
                        for item in cat_results["results"]
                    ]
                    

            else:
                # No date mentioned → just category filter
                cat_results = m.search(
                    user_input,
                    user_id=USER_ID,
                    limit=8,
                    filters={"category": category}
                )
                if cat_results and "results" in cat_results:
                    context_memories = [item["memory"] for item in cat_results["results"]]
                

                 # 🔹 Inject future date into context so the bot can use it in its response
                if date_str and date_type in ["future", "today"]:
                    context_memories.append(f"The user mentioned {date_str} in their message.")

        except Exception as e:
            st.error(f"Memory search error: {e}")

        # Build context string
        if context_memories:
            context_str = "\n".join(context_memories)
        else:
            context_str = "No relevant context found"
        # print(f"retrieved memories: {cat_results}")
        print(f"Memory fetched : {context_str}")

    prompt = build_prompt(context_str, user_input, category, chat_history)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                response = st.session_state.chat_model.generate_content(prompt)
                bot_answer = safe_get_text(response)

                if not bot_answer:
                    bot_answer = "I'm here and listening. Could you tell me a little more?"
                st.markdown(bot_answer)

                # Add to current session
                st.session_state.chat_session.append({
                    "role": "assistant", 
                    "content": bot_answer,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Save to Pinecone

                save_chat(USER_ID, user_input, bot_answer, category, context_str)

            except Exception as e:
                st.error(f"Error generating response: {e}")


    for msg in chat_history:
        ts_display = datetime.fromisoformat(msg['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        print("user_id:", msg['user_name'])
        print("user_message:", msg['user_message'])
        print("bot_response:", msg['bot_response'])
        print("timestamp:", ts_display)
        print("-"*100)

