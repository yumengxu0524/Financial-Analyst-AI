import os
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pinecone
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import httpx

logging.basicConfig(level=logging.INFO)

# HARD-CODED API KEYS (for testing purposes only)
OPENAI_API_KEY = ""
PINECONE_API_KEY = "pcsk_3DH55W_E1PushvzworTJeDsKqZVZu7AvsrBRvxXkhxoGbnaJG2JPzuZFZJkSckFrNaQLvV"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
# Retain the original index name.
CARD_INDEX_NAME = "credit-card-index"

# Create a Pinecone instance without calling pinecone.init()
pc = Pinecone(api_key=PINECONE_API_KEY)

def initialize_pinecone_client(index_name):
    """Retrieve the existing Pinecone index."""
    try:
        if index_name not in pc.list_indexes().names():
            logging.error(f"Index '{index_name}' does not exist. Please run the indexing script first.")
            return None
        return pc.Index(index_name)
    except Exception as e:
        logging.error(f"Error retrieving the index {index_name}: {e}")
        return None

card_index = initialize_pinecone_client(CARD_INDEX_NAME)
if card_index is None:
    raise ValueError("Pinecone index for cards is not available. Please index the data first.")

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return None

async def generate_embeddings_batch(sentences):
    try:
        logging.info(f"Generating embeddings for sentences: {sentences[:2]} ...")
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(
                pool,
                lambda: openai_client.embeddings.create(input=sentences, model="text-embedding-ada-002")
            )
        embeddings = [np.array(item.embedding) for item in response.data]
        return embeddings
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return [np.zeros(1536) for _ in sentences]

def card_summary_from_metadata(md):
    """Generate a short summary from a card's metadata."""
    summary = (
        f"Card: {md.get('cardName', 'N/A')} (Key: {md.get('cardKey', 'N/A')}, "
        f"Network: {md.get('card_network', 'N/A')}). "
        f"Bonus-to-Annual Fee Ratio: {md.get('bonus_to_annual_fee_ratio', 'N/A')}, "
        f"Effective Reward Rate: {md.get('effective_reward_rate', 'N/A')}, "
        f"Credit Range Score: {md.get('credit_range_score', 'N/A')}."
    )
    return summary

async def query_cards(query_text, top_k=5):
    embeddings = await generate_embeddings_batch([query_text])
    query_vector = embeddings[0].tolist()
    logging.info(f"Query vector (first 10 dims): {query_vector[:10]}")
    try:
        result = card_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        if result.matches:
            logging.info(f"Found {len(result.matches)} matching cards.")
            return result.matches
        else:
            logging.warning("No matching cards found.")
            return []
    except Exception as e:
        logging.error(f"Error querying cards: {e}")
        return []

async def generate_answer(query_text, metric_definitions_file, top_k=5):
    matches = await query_cards(query_text, top_k=top_k)
    if not matches:
        return "I'm sorry, I couldn't find any matching credit card records for your query."

    # Create summaries and flag Uber-related benefits by examining the metadata.
    summaries = []
    uber_flag = False
    for match in matches:
        md = match.metadata
        summaries.append(card_summary_from_metadata(md))
        # Check if the flattened metadata 'benefit' or 'spend_bonus' contains 'uber'
        benefits = md.get("benefit", [])
        spend_bonus = md.get("spend_bonus", [])
        combined_text = " ".join(benefits + spend_bonus).lower()
        if "uber" in combined_text:
            uber_flag = True

    cards_text = "\n".join(summaries)
    definitions = load_json(metric_definitions_file)
    if definitions is None:
        definitions_text = "Metric definitions are unavailable."
    else:
        definitions_text = "Key Metric Definitions:\n"
        for d in definitions[:4]:
            definitions_text += f"- {d['name']}: {d['formula']} ({d['explanation']})\n"

    # Build a system prompt that explicitly instructs the model to consider benefit information.
    if uber_flag:
        uber_note = "Note: At least one card explicitly mentions Uber-related benefits."
    else:
        uber_note = "Note: None of the retrieved cards explicitly mention Uber-related benefits."
    
    system_prompt = (
        "You are a financial analyst specializing in credit card comparisons. "
        "Below are key metric definitions and concise summaries of various credit cards including explicit benefit details, "
        "spend bonus categories, and annual spending information. "
        "Focus especially on the benefit details when answering the query. "
        f"{definitions_text}\n"
        "Card Summaries:\n" + cards_text + "\n"
        + uber_note + "\n"
        "Based on the above information, please provide a concise recommendation that directly addresses the user's query."
    )
    user_prompt = f"User Query: {query_text}"
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        )
        if response.status_code != 200:
            logging.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return "I'm sorry, I encountered an error generating the answer."
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return answer

async def main():
    # Path to your metric definitions JSON file
    definitions_file = "credit_card_data/derived_metrics.json"
    
    # Example query.
    query_text = "which credit card has the best benefit for low credit score customer?"
    
    answer = await generate_answer(query_text, definitions_file, top_k=5)
    print("Final Answer:")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
