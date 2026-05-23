import time
import requests
from typing import List, Optional
from tqdm import tqdm
from neo4j import GraphDatabase, TRUST_ALL_CERTIFICATES
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from neo4j.exceptions import Neo4jError

# %% [markdown]
# **INITIALLIZING NEO4J CONNECTION**

# %%
NEO4J_URI = "#####"
NEO4J_USER = "#####"
NEO4J_PASSWORD = "#####"
NEO4J_DATABASE = "#####"

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    encrypted=True,
    trust=TRUST_ALL_CERTIFICATES,
    notifications_min_severity="OFF",
)
try:
    neo4j_driver.verify_connectivity()
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")

# %% [markdown]
# **AI MODEL CONNECTION**

# %%
import os
import requests
from pathlib import Path

# =============================================================================
# LOCAL LLAMA ENVIRONMENT CONFIGURATION (QWEN)
# =============================================================================

# Default local server URLs:
# - Ollama: "http://localhost:11434/v1"
# - Llama.cpp / vLLM / LM Studio: "http://localhost:8000/v1"
LOCAL_API_BASE = "http://localhost:11434/v1"

# Specify the exact tag of the Qwen model you pulled locally (e.g., "qwen2.5", "qwen2.5:7b")
LOCAL_MODEL_NAME = "qwen2.5:7b"

print(f"Local Llama Environment Initialized.")
print(f"   - Target Base URL: {LOCAL_API_BASE}")
print(f"   - Target Model:    {LOCAL_MODEL_NAME}")

def verify_local_connection():
    """Verify that your local Llama/Ollama server is active and responding."""
    try:
        # Check basic models endpoint
        response = requests.get(f"{LOCAL_API_BASE}/models", timeout=300)
        if response.status_code == 200:
            print("Connection successful! Local server is online.")
            available_models = [m.get("id") for m in response.json().get("data", [])]
            print(f"Available local models: {available_models}")
        else:
            print(f"Server responded with status code: {response.status_code}")
    except Exception as e:
        print(f"Connection failed: Is your local server running at {LOCAL_API_BASE}? Error: {e}")

verify_local_connection()

# %% [markdown]
# **CHAT AND EMBEDDINGS FUNCTIONS**

# %%
import time
import re
import requests

LOCAL_API_BASE = globals().get("LOCAL_API_BASE", "http://localhost:11434/v1")
LOCAL_MODEL_NAME = globals().get("LOCAL_MODEL_NAME", "qwen2.5:7b")
LOCAL_EMBED_MODEL = globals().get("LOCAL_EMBED_MODEL", "qwen3-embedding:latest")

# ---------- Local Embeddings ----------
def embed(texts, model=LOCAL_EMBED_MODEL, max_retries=3):
    """
    Generate embeddings using your local server.
    Handles both a single string or a list of text strings.
    """
    url = f"{LOCAL_API_BASE}/embeddings"
    payload = {
        "model": model,
        "input": texts
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            response_data = response.json()
            
            # Local endpoints can return variations depending on if input was a string or list
            data_records = response_data.get("data", [])
            if isinstance(texts, list):
                return [record["embedding"] for record in data_records]
            else:
                return data_records[0]["embedding"]
                
        except Exception as e:
            print(f"⚠️ Local embedding attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise e

# -------------------------------
# 2. Local Chat Completion (OpenAI-compatible endpoint)
# -------------------------------

def chat(messages, model=LOCAL_MODEL_NAME, temperature=0, config={}, max_retries=3):
    """
    Perform a structured message completion using your local Qwen model.
    No more manual string concatenation—we send the message array directly!
    """
    url = f"{LOCAL_API_BASE}/chat/completions"
    
    # Standard OpenAI Chat Geometry
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    payload.update(config)

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract content from choices structure
            content = response_data["choices"][0]["message"]["content"]
            return content.strip()
            
        except Exception as e:
            print(f"⚠️ Local chat completion attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise e

# -------------------------------
# 3. Local Tool‑calling (Function Calling)
# -------------------------------

def tool_choice(messages, model=LOCAL_MODEL_NAME, temperature=0, tools=[], config={}):
    """
    Perform a chat completion that involves local tool selection.
    Uses native OpenAI-style json payloads directly supported by Qwen models.
    """
    url = f"{LOCAL_API_BASE}/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "tools": tools if tools else None
    }
    payload.update(config)

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        response_json = response.json()
        
        message = response_json["choices"][0]["message"]
        if "tool_calls" in message and message["tool_calls"]:
            return message["tool_calls"]
        return []
    except Exception as e:
        print(f"Local Tool-Call execution failed: {e}")
        return []

# %% [markdown]
# **CLINICAL PROMPTS FOR MODEL**

# %%
MAP_SYSTEM_PROMPT = """
---Role---
You are an expert pediatric dental assistant responding to questions based on clinical text.

---Goal---
Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant clinical information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question.

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"
Do not list more than 5 record ids in a single reference.

---Data tables---
{context_data}
"""

REDUCE_SYSTEM_PROMPT = """
---Role---
You are an expert pediatric dental assistant synthesizing clinical perspectives from multiple dental analysts.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarizing all the reports from the analysts who focused on different parts of the dental dataset.

Note that the analysts' reports provided below are ranked in descending order of importance.
If you don't know the answer or if the provided reports do not contain sufficient information, just say so. Do not make anything up.

The final response should remove all irrelevant information and merge the cleaned information into a comprehensive clinical answer. Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

Preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

---Target response length and format---
{response_type}

---Analyst Reports---
{report_data}
"""

LOCAL_SEARCH_SYSTEM_PROMPT = """
---Role---
You are an expert pediatric dental assistant responding to questions based on clinical knowledge graph data.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length, and incorporating any relevant general dental knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:
"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."
Do not list more than 5 record ids in a single reference.

---Target response length and format---
{response_type}

---Data tables---
{context_data}
"""

def get_map_system_prompt(context):
    return MAP_SYSTEM_PROMPT.replace("{context_data}", str(context))

def get_reduce_system_prompt(report_data, response_type="multiple paragraphs"):
    return REDUCE_SYSTEM_PROMPT.replace("{report_data}", str(report_data)).replace("{response_type}", str(response_type))

def get_local_system_prompt(report_data, response_type="multiple paragraphs"):
    return LOCAL_SEARCH_SYSTEM_PROMPT.replace("{context_data}", str(report_data)).replace("{response_type}", str(response_type))

# %% [markdown]
# **RETRIEVER LOGIC**

# %%
def global_retriever(query: str, rating_threshold: float = 0.0) -> str:
    """Useful for broad, conceptual queries requiring a summary of the entire dental graph."""
    community_data, _, _ = neo4j_driver.execute_query(
        "MATCH (c:__Community__) WHERE c.rating >= $rating RETURN c.summary AS summary",
        database_=NEO4J_DATABASE,
        rating=rating_threshold,
    )
    
    intermediate_results = []
    for community in tqdm(community_data, desc="Processing communities"):
        intermediate_messages = [
            {"role": "system", "content": get_map_system_prompt(community["summary"])},
            {"role": "user", "content": query},
        ]
        intermediate_results.append(chat(intermediate_messages))

    final_messages = [
        {"role": "system", "content": get_reduce_system_prompt(intermediate_results)},
        {"role": "user", "content": query},
    ]
    return chat(final_messages)

local_search_query = """
CALL db.index.vector.queryNodes('entities', $k, $embedding)
YIELD node, score
WITH collect(node) as nodes
WITH collect {
    UNWIND nodes as n
    MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
    WITH c, count(distinct n) as freq
    RETURN c.text AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH c, c.rating as rank
    RETURN c.summary 
    ORDER BY rank DESC
    LIMIT $topCommunities
} AS report_mapping,
collect {
    UNWIND nodes as n
    MATCH (n)-[r:SUMMARIZED_RELATIONSHIP]-(m) 
    WHERE m IN nodes
    RETURN r.summary AS descriptionText
    LIMIT $topInsideRels
} as insideRels,
collect {
    UNWIND nodes as n
    RETURN n.summary AS descriptionText
} as entities
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: insideRels, Entities: entities} AS text
"""

def local_search(query: str, k_entities=5, topChunks=3, topCommunities=3, topInsideRels=3) -> str:
    """Useful for specific queries about distinct dental conditions, symptoms, or treatments."""
    context, _, _ = neo4j_driver.execute_query(
        local_search_query,
        database_=NEO4J_DATABASE,
        embedding=embed(query), 
        topChunks=topChunks,
        topCommunities=topCommunities,
        topInsideRels=topInsideRels,
        k=k_entities,
    )
    context_str = str(context[0]["text"])
    local_messages = [
        {"role": "system", "content": get_local_system_prompt(context_str)},
        {"role": "user", "content": query},
    ]
    return chat(local_messages)

# %% [markdown]
# **FAST API CONNECTOR**

# %%
app = FastAPI(title="Dental Knowledge Retriever API")

class QueryRequest(BaseModel):
    query: str
    search_type: str = "local"

@app.post("/ask")
async def ask_bot(request: QueryRequest):
    try:
        if request.search_type == "global":
            answer = global_retriever(request.query)
        else:
            answer = local_search(request.query)
        return {"response": answer, "status": "success"}
    except Neo4jError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j service unavailable: {e}",
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local model service unavailable. "
                f"Check that {LOCAL_API_BASE} is running and reachable. Error: {e}"
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5501)


