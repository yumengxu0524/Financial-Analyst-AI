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
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Path to your full data source
ALL_CARD_DATA_FILE = "credit_card_data/all_card_info.json"

def load_all_cards_data(file_path):
    """Load and flatten the full credit card dataset from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Loaded data from {file_path}")
        # Assuming your file structure is a list of lists:
        all_cards = []
        for sublist in data:
            all_cards.extend(sublist)
        return all_cards
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return []

def filter_cards_by_keys(all_cards, card_keys):
    """Return only the cards whose 'cardKey' is in the provided list."""
    filtered = [card for card in all_cards if card.get("cardKey") in card_keys]
    logging.info(f"Filtered down to {len(filtered)} cards for keys: {card_keys}")
    return filtered

def card_to_detailed_text(record):
    """
    Convert a card record into a detailed text snippet that includes all core data.
    This version is designed to be as complete as possible.
    """
    meta = record.get("metadata", record)
    
    header = f"Card: {meta.get('cardName', 'Unknown')} (Key: {meta.get('cardKey', 'N/A')})"
    
    basic_info = (
        f"Network: {meta.get('cardNetwork', meta.get('card_network', 'N/A'))}; "
        f"Annual Fee: {meta.get('annualFee', 'N/A')}; "
        f"Credit Range: {meta.get('creditRange', 'N/A')}"
    )
    
    # Detailed Benefits: use full benefit details.
    benefits = record.get("benefit", [])
    if benefits:
        benefit_details = []
        for b in benefits:
            title = b.get("benefitTitle", "").strip()
            desc = b.get("benefitDesc", "").strip()
            benefit_details.append(f"{title}: {desc}")
        benefits_text = "Detailed Benefits: " + " | ".join(benefit_details)
    else:
        benefits_text = "Detailed Benefits: None provided"
    
    # Spend Bonus Categories details.
    spend_bonus = record.get("spendBonusCategory", [])
    if spend_bonus:
        bonus_details = []
        for sb in spend_bonus:
            if isinstance(sb, str):
                bonus_details.append(sb)
            elif isinstance(sb, dict):
                name = sb.get("spendBonusCategoryName", "").strip()
                multiplier = sb.get("earnMultiplier", "")
                desc = sb.get("spendBonusDesc", "").strip()
                bonus_details.append(f"{name} ({multiplier}x): {desc}")
        bonus_text = "Spend Bonus Categories: " + " | ".join(bonus_details)
    else:
        bonus_text = "Spend Bonus Categories: None provided"
    
    # Annual Spend details.
    annual_spend = record.get("annualSpend", [])
    if annual_spend:
        annual_spend_text = "Annual Spend: " + ", ".join(str(s) for s in annual_spend)
    else:
        annual_spend_text = "Annual Spend: None"
    
    full_text = "\n".join([header, basic_info, benefits_text, bonus_text, annual_spend_text])
    return full_text

def build_cards_summary(cards):
    """
    Given a list of card records, build a combined summary text.
    """
    summaries = []
    for card in cards:
        summaries.append(card_to_detailed_text(card))
    return "\n\n".join(summaries)

def load_json(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return None
    
async def generate_answer_for_selected_cards(user_query, card_keys, metric_definitions_file):
    """
    Generate an answer based on the selected credit cards.
    The user provides the list of cards (by key) they have.
    """
    # Load full data and filter for the selected cards.
    all_cards = load_all_cards_data(ALL_CARD_DATA_FILE)
    selected_cards = filter_cards_by_keys(all_cards, card_keys)
    if not selected_cards:
        return "I'm sorry, I couldn't find any matching credit card records for the specified keys."
    
    # Build a detailed summary of the selected cards.
    cards_summary = build_cards_summary(selected_cards)
    
    # Load metric definitions if needed.
    definitions = load_json(metric_definitions_file)
    if definitions:
        definitions_text = "Key Metric Definitions:\n" + "\n".join(
            f"- {d['name']}: {d['formula']} ({d['explanation']})" for d in definitions[:4]
        )
    else:
        definitions_text = "Metric definitions are unavailable."
    
    # Construct a system prompt that includes the detailed summaries.
    system_prompt = (
        "You are a financial analyst specializing in credit card comparisons. "
        "Below are detailed summaries of the credit cards the user has provided, including explicit benefit information, "
        "spend bonus categories, and annual spending details. "
        "When answering the user's query, use the information from these summaries to provide a recommendation that directly addresses the query. "
        "Definitions of key metrics for reference:\n" + definitions_text + "\n"
        "Card Summaries:\n" + cards_summary + "\n"
        "Based on the above, please provide a concise, benefit-focused recommendation that directly addresses the user's query."
    )
    
    user_prompt = f"User Query: {user_query}"
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4",  # Or choose another supported model.
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
    # Example: The user specifies the cards they have.
    # For example, the user has "amex-gold" and "usbank-cash365".
    user_card_keys = ["amex-gold", "capitalone-quicksilver ", "Discover it cash back"]
    # Example query:
    user_query = "which credit card is the best for travel?"
    
    definitions_file = "credit_card_data/derived_metrics.json"
    answer = await generate_answer_for_selected_cards(user_query, user_card_keys, definitions_file)
    print("Final Answer:")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())