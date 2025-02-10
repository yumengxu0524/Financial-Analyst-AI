# ai_bid_player.py

import os
import json
import logging
import asyncio
import httpx
import re
import html
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO)


# ---------------------------
# PyTorch Model for Bid Rate
# ---------------------------
class BidRateModel(nn.Module):
    def __init__(self, num_categories: int, initial_rates: List[float]):
        super(BidRateModel, self).__init__()
        # Embedding layer outputs 1-dimensional bid rates.
        self.embedding = nn.Embedding(num_categories, 1)
        with torch.no_grad():
            weight_tensor = torch.tensor(initial_rates, dtype=torch.float).unsqueeze(1)
            self.embedding.weight.copy_(weight_tensor)

    def forward(self, category_index: torch.Tensor) -> torch.Tensor:
        rate = self.embedding(category_index)  # shape (batch_size, 1)
        return rate.squeeze(1)  # shape (batch_size,)


# ---------------------------
# Agent 10: Bidding Agent with Self-Training via PyTorch
# ---------------------------
class AIAgent10:
    """
    Agent 10 computes a bid for each transaction based on a dynamic bid rate model.
    It self-adjusts its bid rate using a simple MSE update and also queries OpenAI
    to extract competitor recommendation text (if needed).
    """
    def __init__(self, OPENAI_API_KEY: str, initial_budget: float):
        self.api_key = OPENAI_API_KEY
        self.budget = initial_budget  # total reward budget
        
        # Define categories and initial bid rates.
        self.categories = [
            "groceries", "restaurant", "gas", "uber", "travel",
            "utilities", "entertainment", "online shopping", "cellphone", "health"
        ]
        self.category_to_index = {cat: i for i, cat in enumerate(self.categories)}
        initial_rates = [0.05, 0.04, 0.03, 0.02, 0.03, 0.02, 0.03, 0.03, 0.02, 0.02]
        num_categories = len(self.categories)
        
        # Create the PyTorch model and optimizer.
        self.bid_model = BidRateModel(num_categories, initial_rates)
        self.optimizer = optim.Adam(self.bid_model.parameters(), lr=0.01)
        
        # Competitor strength per category (baseline 1.0).
        self.competitor_strength = {}

    def sanitize_input(self, input_str: str) -> str:
        return html.unescape(input_str).strip()

    async def generate_answer_for_selected_cards(self, user_query: str, card_keys: List[str]) -> str:
        """
        (Optional helper) Query OpenAI to get some analysis.
        Here it is used to add text to our explanation.
        """
        system_prompt = (
            "You are a financial analyst specializing in credit card comparisons. "
            "Please analyze the query below and provide a concise recommendation, ending your answer "
            "with a new line exactly as: \nRecommended Card Key: <card_key>\n, where <card_key> is one of the provided keys."
        )
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": self.sanitize_input(user_query)}
                        ]
                    }
                )
            if response.status_code != 200:
                logging.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return "I'm sorry, I encountered an error generating the answer."
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            logging.error(f"Exception during OpenAI API call: {e}")
            return "I'm sorry, I encountered an exception during processing."

    async def bid_on_transaction(self, transaction: dict, remaining_count: int, competitor_keys: List[str]) -> dict:
        """
        Compute a bid for a single transaction.
        Bid = transaction amount × bid rate (from model), capped by the allowed bid.
        """
        merchant = self.sanitize_input(transaction.get("merchant", "unknown merchant"))
        transaction["description"] = self.sanitize_input(transaction.get("description", ""))
        category = transaction.get("category", "").lower()
        amount = transaction.get("amount", 0)
        
        predicted_rate = self.bid_rate(category)
        heuristic_bid = amount * predicted_rate

        # Adjust allowed bid using competitor strength (default 1.0).
        strengths = [self.competitor_strength.get(category, {}).get(comp, 1.0) for comp in competitor_keys]
        adjustment_factor = max(strengths) if strengths else 1.0

        allowed_bid = (self.budget / remaining_count) / adjustment_factor if remaining_count > 0 else self.budget
        bid = min(heuristic_bid, allowed_bid)
        self.budget -= bid

        explanation = (
            f"For a ${amount} transaction in '{category}', our model predicts a bid rate of {predicted_rate*100:.1f}%, "
            f"yielding a heuristic bid of ${heuristic_bid:.2f}. With the current budget, the allowed bid (after adjustment "
            f"for competitor strength {adjustment_factor:.2f}) is ${allowed_bid:.2f}. Final bid: ${bid:.2f}. "
            f"Remaining budget: ${self.budget:.2f}."
        )
        recommended = competitor_keys[0] if competitor_keys else "unknown"
        return {
            "transaction": transaction,
            "bid": bid,
            "allowed_bid": allowed_bid,
            "recommended_card": recommended,
            "explanation": explanation
        }

    def bid_rate(self, category: str) -> float:
        index = self.category_to_index.get(category.lower())
        if index is None:
            return 0.02
        self.bid_model.eval()
        with torch.no_grad():
            rate = self.bid_model(torch.tensor([index]))
        return rate.item()

    def update_model_for_category(self, category: str, target_rate: float):
        index = self.category_to_index.get(category.lower())
        if index is None:
            return
        self.bid_model.train()
        input_tensor = torch.tensor([index])
        predicted_rate = self.bid_model(input_tensor)
        target_tensor = torch.tensor([target_rate], dtype=torch.float)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted_rate, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        new_rate = self.bid_model(torch.tensor([index])).item()
        logging.info(f"Updated model for '{category}': new rate = {new_rate:.4f}, loss = {loss.item():.6f}")

    async def process_transaction(self, transaction: dict, competitor_keys: List[str], remaining_count: int) -> dict:
        merchant = self.sanitize_input(transaction.get("merchant", "unknown merchant"))
        category = transaction.get("category", "").lower()
        amount = transaction.get("amount", 0)
        user_query = (
            f"For a transaction of ${amount} at {merchant} in the {transaction['category']} category "
            f"(description: {transaction['description']}), which credit card among the following would yield the highest rewards? "
            "Please provide the recommended card key and a brief explanation."
        )
        # (Optional) get extra text from OpenAI.
        answer = await self.generate_answer_for_selected_cards(user_query, competitor_keys)
        match = re.search(r"Recommended Card Key:\s*(\S+)", answer, re.IGNORECASE)
        recommended = match.group(1) if match else (competitor_keys[0] if competitor_keys else "unknown")

        bid_info = await self.bid_on_transaction(transaction, remaining_count, competitor_keys)
        bid_info["explanation"] += "\n\n" + answer
        bid_info["recommended_card"] = recommended

        predicted_rate = self.bid_rate(category)
        heuristic_bid = amount * predicted_rate
        allowed_bid = bid_info.get("allowed_bid", heuristic_bid)
        target_rate = allowed_bid / amount if allowed_bid < heuristic_bid else predicted_rate

        self.update_model_for_category(category, target_rate)
        return bid_info

    async def process_transactions(self, transactions: list, competitor_keys: List[str]) -> list:
        results = []
        total = len(transactions)
        for i, tx in enumerate(transactions):
            remaining_count = total - i
            res = await self.process_transaction(tx, competitor_keys, remaining_count)
            results.append(res)
        return results

    def save_scores(self, scores: dict, filename: str = None):
        if filename is None:
            filename = "agent10_transaction_scores.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=4)
            logging.info(f"Saved scores to {filename}")
        except Exception as e:
            logging.error(f"Error saving scores to {filename}: {e}")


# ---------------------------
# Testing Agent 10 (if run directly)
# ---------------------------
if __name__ == "__main__":
    import asyncio
    agent10 = AIAgent10(OPENAI_API_KEY="your-openai-api-key-here", initial_budget=20.0)
    transactions = [
        {"category": "groceries", "amount": 150, "description": "$150 at Trader Joe's", "merchant": "Trader Joe's"},
        {"category": "groceries", "amount": 200, "description": "$200 at Costco", "merchant": "Costco"},
        {"category": "restaurant", "amount": 75, "description": "$75 dinner at a local restaurant", "merchant": "Local Restaurant"}
    ]
    competitor_keys = ["capitalone-quicksilver", "discover-cashback", "chase-freedom", "usaa-cashbackrewardsplusamex"]
    results = asyncio.run(agent10.process_transactions(transactions, competitor_keys))
    print(json.dumps(results, indent=2))
