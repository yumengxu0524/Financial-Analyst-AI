# Agent_9_1_competitor_offer.py

import json
import logging
import html
from typing import List

logging.basicConfig(level=logging.INFO)

# File paths (keep ALL_CARD_DATA_FILE unchanged)
ALL_CARD_DATA_FILE = "credit_card_data/all_card_info.json"
METRIC_DEFINITIONS_FILE = "credit_card_data/derived_metrics.json"
RESTRUCTURED_DATA_FILE = "credit_card_data/card_metrics_restructured.json"
SCORES_OUTPUT_FILE = "transaction_scores.json"


# ------------------------------
# DATA LOADING FUNCTIONS
# ------------------------------

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
    """Load and flatten the full credit card dataset from the raw file."""
    data = load_json(file_path)
    if not data:
        return []
    all_cards = []
    # Flatten if file contains a list of lists.
    for sublist in data:
        if isinstance(sublist, list):
            all_cards.extend(sublist)
        else:
            all_cards.append(sublist)
    return all_cards

def load_restructured_data(file_path):
    """Load the restructured card data."""
    data = load_json(file_path)
    if not data:
        return []
    return data


# ------------------------------
# DATA MERGING & FILTERING FUNCTIONS
# ------------------------------

def merge_records(raw, restructured):
    """
    Merge two card records (raw and restructured) based on cardKey.
    Values from the restructured data (e.g., derived metrics) take precedence.
    """
    merged = raw.copy()
    raw_meta = raw.get("metadata", {})
    restruct_meta = restructured.get("metadata", {})
    merged_meta = raw_meta.copy()
    merged_meta.update(restruct_meta)
    merged["metadata"] = merged_meta

    if "derived_metrics" in restructured:
        merged["derived_metrics"] = restructured["derived_metrics"]
    merged["benefit"] = raw.get("benefit", [])
    merged["spendBonusCategory"] = raw.get("spendBonusCategory", [])
    merged["annualSpend"] = raw.get("annualSpend", [])
    return merged

def merge_card_data(raw_cards, restructured_cards):
    """
    Merge raw card data and restructured data based on cardKey.
    """
    merged_dict = {}
    for card in raw_cards:
        key = card.get("cardKey")
        if key:
            merged_dict[key] = card
    for card in restructured_cards:
        key = card.get("metadata", {}).get("cardKey")
        if key:
            if key in merged_dict:
                merged_dict[key] = merge_records(merged_dict[key], card)
            else:
                merged_dict[key] = card
    return list(merged_dict.values())

def filter_cards_by_keys(cards, card_keys):
    """Return only the cards whose cardKey is in the provided list."""
    filtered = [card for card in cards if card.get("cardKey") in card_keys or 
                (card.get("metadata", {}).get("cardKey") in card_keys)]
    logging.info(f"Filtered down to {len(filtered)} cards for keys: {card_keys}")
    return filtered


# ------------------------------
# SUMMARY GENERATION FUNCTIONS
# ------------------------------

def card_to_detailed_text(record):
    """
    Create a detailed text summary for a card record.
    Includes card details, explicit benefit information, spend bonus categories,
    annual spend, and derived metric values.
    """
    meta = record.get("metadata", {})
    derived = record.get("derived_metrics", {})
    benefits = record.get("benefit", [])
    spend_bonus = record.get("spendBonusCategory", [])
    annual_spend = record.get("annualSpend", [])

    header = f"Card: {meta.get('cardName', 'Unknown')} (Key: {meta.get('cardKey', 'N/A')})"
    basic_info = (
        f"Network: {meta.get('cardNetwork', meta.get('card_network', 'N/A'))}; "
        f"Annual Fee: {meta.get('annualFee', 'N/A')}; "
        f"Credit Range: {meta.get('creditRange', 'N/A')}"
    )

    if benefits:
        benefit_details = []
        for b in benefits:
            title = b.get("benefitTitle", "").strip()
            desc = b.get("benefitDesc", "").strip()
            benefit_details.append(f"{title}: {desc}")
        benefits_text = "Detailed Benefits: " + " | ".join(benefit_details)
    else:
        benefits_text = "Detailed Benefits: None provided"

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

    if annual_spend:
        annual_spend_text = "Annual Spend: " + ", ".join(str(s) for s in annual_spend)
    else:
        annual_spend_text = "Annual Spend: None"

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
    """Load metric definitions (explanation file) and return a text summary."""
    data = load_json(file_path)
    if not data:
        return "No metric definitions available."
    definitions = []
    for item in data:
        definitions.append(f"{item.get('name', 'N/A')}: {item.get('formula', 'N/A')} ({item.get('explanation', 'No explanation')})")
    return "\n".join(definitions)


# ------------------------------
# COMPETITOR OFFERS PREPARATION
# ------------------------------

class OfferWinerAgent:
    """
    This agent prepares competitor credit card offers.
    It leverages the merged card data to build a detailed text summary
    (including metric definitions and card details) that the Judge Agent can include
    in its prompt for evaluation.
    """
    def __init__(self, OPENAI_API_KEY: str):
        self.api_key = OPENAI_API_KEY

    def get_competitor_offers_text(self, card_keys: List[str]) -> str:
        # Load raw and restructured card data.
        raw_cards = load_all_cards_data(ALL_CARD_DATA_FILE)
        restructured_cards = load_restructured_data(RESTRUCTURED_DATA_FILE)
        # Merge the two data sources.
        merged_cards = merge_card_data(raw_cards, restructured_cards)
        # Filter the cards by the provided competitor card keys.
        filtered_cards = filter_cards_by_keys(merged_cards, card_keys)
        # Build a detailed summary of the competitor cards.
        summary = build_cards_summary(filtered_cards)
        # Load metric definitions.
        definitions = load_metric_definitions(METRIC_DEFINITIONS_FILE)
        # Compose the full competitor offers text.
        offers_text = (
            f"Metric Definitions:\n{definitions}\n\n"
            f"Competitor Credit Card Offers:\n{summary}"
        )
        return offers_text


# ---------------------------
# For testing OfferWinerAgent (if run directly)
# ---------------------------
if __name__ == "__main__":
    card_keys = ["card1-key", "card2-key", "card3-key", "card4-key"]
    agent = OfferWinerAgent(OPENAI_API_KEY="your-openai-api-key-here")
    offers_text = agent.get_competitor_offers_text(card_keys)
    print(offers_text)
