import os
import json
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

# HARD-CODED API KEYS (for testing purposes only)
OPENAI_API_KEY = ""  # Replace with your actual API key

# IMPORTANT: Keep this file name unchanged.
ALL_CARD_DATA_FILE = "credit_card_data/all_card_info.json"

# (Optional) The file with metric definitions/explanations.
METRIC_DEFINITIONS_FILE = "credit_card_data/derived_metrics.json"
# (Optional) The file with restructured metrics.
RESTRUCTURED_DATA_FILE = "credit_card_data/card_metrics_restructured.json"

# Initialize OpenAI client.
openai_client = OpenAI(api_key=OPENAI_API_KEY)

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

def load_all_cards_data(file_path):
    """Load and flatten the full credit card dataset from the specified file."""
    data = load_json(file_path)
    if not data:
        return []
    all_cards = []
    # If the file contains a list of lists, flatten it.
    for sublist in data:
        if isinstance(sublist, list):
            all_cards.extend(sublist)
        else:
            all_cards.append(sublist)
    return all_cards

def filter_cards_by_keys(cards, card_keys):
    """Return only the cards whose 'cardKey' is in the provided list."""
    filtered = [card for card in cards if card.get("cardKey") in card_keys]
    logging.info(f"Filtered down to {len(filtered)} cards for keys: {card_keys}")
    return filtered

def card_to_detailed_text(record):
    """
    Convert a card record into a detailed text summary that includes:
      - Basic information (card name, key, network, annual fee, credit range)
      - Detailed benefits (titles and descriptions)
      - Spend bonus categories (names, multipliers, descriptions)
      - Annual spend details (if available)
      - Derived metrics values (if available in the record)
    """
    meta = record.get("metadata", record)
    derived = record.get("derived_metrics", {})
    benefits = record.get("benefit", [])
    spend_bonus = record.get("spendBonusCategory", [])
    annual_spend = record.get("annualSpend", [])

    # Header with clear identification.
    header = f"Card: {meta.get('cardName', 'Unknown')} (Key: {meta.get('cardKey', 'N/A')})"

    # Basic details.
    basic_info = (
        f"Network: {meta.get('cardNetwork', meta.get('card_network', 'N/A'))}; "
        f"Annual Fee: {meta.get('annualFee', 'N/A')}; "
        f"Credit Range: {meta.get('creditRange', 'N/A')}"
    )

    # Detailed Benefits.
    if benefits:
        benefit_details = []
        for b in benefits:
            title = b.get("benefitTitle", "").strip()
            desc = b.get("benefitDesc", "").strip()
            benefit_details.append(f"{title}: {desc}")
        benefits_text = "Detailed Benefits: " + " | ".join(benefit_details)
    else:
        benefits_text = "Detailed Benefits: None provided"

    # Spend Bonus Categories.
    if spend_bonus:
        bonus_details = []
        for sb in spend_bonus:
            if isinstance(sb, dict):
                name = sb.get("spendBonusCategoryName", "").strip()
                multiplier = sb.get("earnMultiplier", "")
                desc = sb.get("spendBonusDesc", "").strip()
                bonus_details.append(f"{name} ({multiplier}x): {desc}")
            elif isinstance(sb, str):
                bonus_details.append(sb)
        bonus_text = "Spend Bonus Categories: " + " | ".join(bonus_details)
    else:
        bonus_text = "Spend Bonus Categories: None provided"

    # Annual Spend.
    if annual_spend:
        annual_spend_text = "Annual Spend: " + ", ".join(str(s) for s in annual_spend)
    else:
        annual_spend_text = "Annual Spend: None"

    # Derived Metrics (if any are present in the record).
    if derived:
        metric_details = []
        for k, v in derived.items():
            metric_details.append(f"{k}: {v}")
        metrics_text = "Derived Metrics: " + " | ".join(metric_details)
    else:
        metrics_text = "Derived Metrics: None"

    full_text = "\n".join([header, basic_info, benefits_text, bonus_text, annual_spend_text, metrics_text])
    return full_text

def build_cards_summary(cards):
    """Combine the detailed summaries of the selected cards into one text block."""
    summaries = [card_to_detailed_text(card) for card in cards]
    return "\n\n".join(summaries)

def load_metric_definitions(file_path):
    """Load metric definitions from the provided JSON file (used for explanation)."""
    data = load_json(file_path)
    if not data:
        return "No metric definitions available."
    definitions = []
    for item in data:
        definitions.append(f"{item.get('name', 'N/A')}: {item.get('formula', 'N/A')} ({item.get('explanation', 'No explanation')})")
    return "\n".join(definitions)

async def generate_answer_for_selected_cards(user_query, card_keys):
    """
    Generate an answer based on the selected cards.
    The prompt includes detailed card summaries from ALL_CARD_DATA_FILE and metric definitions.
    """
    # Load full card data.
    all_cards = load_all_cards_data(ALL_CARD_DATA_FILE)
    selected_cards = filter_cards_by_keys(all_cards, card_keys)
    if not selected_cards:
        return "I'm sorry, I couldn't find any matching credit card records for the specified keys."
    
    cards_summary = build_cards_summary(selected_cards)
    
    # Load metric definitions (explanation file).
    metric_definitions = load_metric_definitions(METRIC_DEFINITIONS_FILE)
    
    # Construct the system prompt.
    system_prompt = (
        "You are a financial analyst specializing in credit card comparisons. "
        "Below are detailed summaries of the credit cards the user has provided. "
        "Each summary includes comprehensive information: card details, explicit benefit information, spend bonus categories, annual spend, and derived metric values. "
        "Metric definitions for reference are provided below:\n\n" + metric_definitions + "\n\n"
        "Card Summaries:\n" + cards_summary + "\n\n"
        "Based on the above information, please provide a concise, benefit-focused recommendation or analysis that directly addresses the user's query."
    )
    
    user_prompt = f"User Query: {user_query}"
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4",  # or another supported model
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
    # Example: the user specifies which cards they have.
    user_card_keys = ["amex-gold", "Capitalone-Venture", "chase-sapphire-preferred"]
    # Example query:
    user_query = "I am planning a hawaii trip with airline tickets and hotels which cards are suitable?"
    
    answer = await generate_answer_for_selected_cards(user_query, user_card_keys)
    print("Final Answer:")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
