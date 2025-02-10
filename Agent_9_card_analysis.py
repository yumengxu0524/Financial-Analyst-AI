import re
import json
import logging
import asyncio
import httpx
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor  # (currently unused)

logging.basicConfig(level=logging.INFO)


class OfferWinerAgent:
    def __init__(self, OPENAI_API_KEY: str):
        self.api_key = OPENAI_API_KEY

        # File paths (do not change ALL_CARD_DATA_FILE)
        self.ALL_CARD_DATA_FILE = "credit_card_data/all_card_info.json"
        self.METRIC_DEFINITIONS_FILE = "credit_card_data/derived_metrics.json"
        self.RESTRUCTURED_DATA_FILE = "credit_card_data/card_metrics_restructured.json"
        self.SCORES_OUTPUT_FILE = "transaction_scores.json"

    def load_json(self, file_path: str):
        """Load JSON data from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info(f"Loaded data from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading JSON from {file_path}: {e}")
            return None

    def load_all_cards_data(self, file_path: str):
        """Load and flatten the full credit card dataset from the raw file."""
        data = self.load_json(file_path)
        if not data:
            return []
        all_cards = []
        # Flatten if the file contains a list of lists.
        for sublist in data:
            if isinstance(sublist, list):
                all_cards.extend(sublist)
            else:
                all_cards.append(sublist)
        return all_cards

    def merge_records(self, raw: dict, restructured: dict) -> dict:
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

        # Keep original benefit, spendBonusCategory, and annualSpend fields.
        merged["benefit"] = raw.get("benefit", [])
        merged["spendBonusCategory"] = raw.get("spendBonusCategory", [])
        merged["annualSpend"] = raw.get("annualSpend", [])
        return merged

    def merge_card_data(self, raw_cards: list, restructured_cards: list) -> list:
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
                    merged_dict[key] = self.merge_records(merged_dict[key], card)
                else:
                    merged_dict[key] = card
        return list(merged_dict.values())

    def filter_cards_by_keys(self, cards: list, card_keys: list) -> list:
        """Return only the cards whose cardKey is in the provided list."""
        filtered = [
            card for card in cards
            if card.get("cardKey") in card_keys or (card.get("metadata", {}).get("cardKey") in card_keys)
        ]
        logging.info(f"Filtered down to {len(filtered)} cards for keys: {card_keys}")
        return filtered

    def card_to_detailed_text(self, record: dict) -> str:
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
            metric_details = [f"{k}: {v}" for k, v in derived.items()]
            metrics_text = "Derived Metrics: " + " | ".join(metric_details)
        else:
            metrics_text = "Derived Metrics: None"

        full_text = "\n".join(
            [header, basic_info, benefits_text, bonus_text, annual_spend_text, metrics_text]
        )
        return full_text

    def build_cards_summary(self, cards: list) -> str:
        """Combine the detailed summaries of the selected cards into one text block."""
        summaries = [self.card_to_detailed_text(card) for card in cards]
        return "\n\n".join(summaries)

    def load_metric_definitions(self, file_path: str) -> str:
        """Load metric definitions (explanation file) and return a text summary."""
        data = self.load_json(file_path)
        if not data:
            return "No metric definitions available."
        definitions = [
            f"{item.get('name', 'N/A')}: {item.get('formula', 'N/A')} ({item.get('explanation', 'No explanation')})"
            for item in data
        ]
        return "\n".join(definitions)

    async def generate_answer_for_selected_cards(self, user_query, card_keys):
        raw_cards = self.load_all_cards_data(self.ALL_CARD_DATA_FILE)
        restructured_cards = self.load_json(self.RESTRUCTURED_DATA_FILE)
        if not raw_cards:
            return "I'm sorry, no raw card data is available."
        if not restructured_cards:
            return "I'm sorry, no restructured card data is available."

        merged_cards = self.merge_card_data(raw_cards, restructured_cards)
        selected_cards = self.filter_cards_by_keys(merged_cards, card_keys)
        if not selected_cards:
            return "I'm sorry, I couldn't find any matching credit card records for the specified keys."

        cards_summary = self.build_cards_summary(selected_cards)
        metric_definitions = self.load_metric_definitions(self.METRIC_DEFINITIONS_FILE)

        # IMPORTANT: Instruct the AI to output a final line that exactly states the recommended card key.
        system_prompt = (
            "You are a financial analyst specializing in credit card comparisons. "
            "Below are detailed summaries of the credit cards the user has provided. "
            "Each summary includes comprehensive information: card details, explicit benefit information, "
            "spend bonus categories, annual spend, and derived metric values. "
            "Metric definitions for reference are provided below:\n\n" +
            metric_definitions + "\n\n" +
            "Card Summaries:\n" + cards_summary + "\n\n" +
            "Based on the above information, please provide a concise, benefit-focused recommendation or analysis that directly addresses the user's query. "
            "At the very end of your answer, on a new line, output exactly: \n"
            "Recommended Card Key: <card_key>\n"
            "Where <card_key> must be one of the provided card keys and must exactly match the recommended card mentioned in the explanation. "
            "Do not output any other card key in that line."
        )

        user_prompt = f"User Query: {user_query}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
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

    async def process_transaction(self, transaction: dict, card_keys: list) -> dict:
        merchant = transaction.get("merchant", "unknown merchant")
        user_query = (
            f"For a transaction of ${transaction['amount']} at {merchant} in the {transaction['category']} category "
            f"(description: {transaction['description']}), which credit card among the following would yield the highest rewards? "
            "Please provide the recommended card key and a brief explanation."
        )
        answer = await self.generate_answer_for_selected_cards(user_query, card_keys)

        # Use regex to extract the recommended card key.
        match = re.search(r"Recommended Card Key:\s*(\S+)", answer, re.IGNORECASE)
        if match:
            recommended = match.group(1)
        else:
            # Fallback: naive extraction if explicit output was not found.
            recommended = None
            for key in card_keys:
                if key.lower() in answer.lower():
                    recommended = key
                    break
            if not recommended:
                recommended = "unknown"

        return {
            "transaction": transaction,
            "recommended_card": recommended,
            "amount": transaction["amount"],
            "explanation": answer
        }


    async def process_transactions(self, transactions: list, card_keys: list) -> list:
        """Process a list of transactions asynchronously."""
        results = []
        for tx in transactions:
            res = await self.process_transaction(tx, card_keys)
            results.append(res)
        return results

    def save_scores(self, scores: dict, filename: str = None):
        """Save the aggregated scores (a dict) to a JSON file."""
        if filename is None:
            filename = self.SCORES_OUTPUT_FILE
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=4)
            logging.info(f"Saved scores to {filename}")
        except Exception as e:
            logging.error(f"Error saving scores to {filename}: {e}")

    def generate_plot_file(self, scores: dict, filename: str = "plot.png"):
        """
        Generate a pie chart of total transaction amounts per recommended card,
        save it to a file, and return the file name.
        """
        import matplotlib.pyplot as plt

        labels = list(scores.keys())
        sizes = list(scores.values())

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Credit Card Consumption Share by Recommended Card")
        plt.savefig(filename)
        plt.close()
        logging.info(f"Plot saved to {filename}")
        return filename
