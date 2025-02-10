# judge_agent.py

import json
import logging
import asyncio
import httpx
from typing import List

logging.basicConfig(level=logging.INFO)

class OfferWinerAgentJudge:
    """
    This judge evaluates a transaction’s bids.
    It receives Agent 10’s bid info and a detailed competitor offers text,
    then constructs a prompt for ChatGPT to decide which offer wins.
    """
    def __init__(self, OPENAI_API_KEY: str):
        self.api_key = OPENAI_API_KEY

    async def generate_judgement(self, transaction: dict, agent10_bid_info: dict, competitor_offers_text: str) -> dict:
        agent10_bid = agent10_bid_info.get("bid", 0)
        agent10_card = agent10_bid_info.get("recommended_card", "unknown")
        
        prompt = (
            "You are a bidding judge. A transaction is up for bidding with the following details:\n"
            f"Category: {transaction.get('category')}\n"
            f"Amount: ${transaction.get('amount')}\n"
            f"Description: {transaction.get('description')}\n"
            f"Merchant: {transaction.get('merchant')}\n\n"
            "Agent 10's bid:\n"
            f"Bid Amount: ${agent10_bid:.2f}\n"
            f"Recommended Card: {agent10_card}\n\n"
            "Competitor Credit Card Offers (detailed, with reward rates expressed as percentages):\n"
            f"{competitor_offers_text}\n\n"
            "Based on the above, if Agent 10's bid is greater than or equal to the best competitor offer, "
            "then Agent 10 wins; otherwise, the competitor with the best offer wins. "
            "Return your answer in JSON format with exactly these two keys:\n"
            '{"result": "<win/lose>", "winner": "<winning bidder>"}'
        )
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {"role": "system", "content": "You are a bidding judge."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )
            if response.status_code != 200:
                logging.error(f"OpenAI API error in judgement: {response.status_code} - {response.text}")
                return {"result": "error", "winner": "unknown"}
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            try:
                judgement = json.loads(answer)
            except Exception as e:
                logging.error(f"Error parsing judgement JSON: {e}. Raw answer: {answer}")
                judgement = {"result": "error", "winner": "unknown"}
            return judgement
        except Exception as e:
            logging.error(f"Exception during OpenAI API call in judgement: {e}")
            return {"result": "error", "winner": "unknown"}

    async def process_transaction(self, transaction: dict, agent10_bid_info: dict, competitor_offers_text: str) -> dict:
        judgement = await self.generate_judgement(transaction, agent10_bid_info, competitor_offers_text)
        return judgement
    

    async def process_transactions(self, transactions: list, agent10_bid_list: list, competitor_offers_text: str) -> list:
        results = []
        for tx, bid_info in zip(transactions, agent10_bid_list):
            res = await self.process_transaction(tx, bid_info, competitor_offers_text)
            results.append(res)
        return results


# ---------------------------
# For testing Judge Agent (if run directly)
# ---------------------------
if __name__ == "__main__":
    import asyncio
    API_KEY = "your-openai-api-key-here"
    judge = OfferWinerAgentJudge(API_KEY)
    transaction = {"category": "groceries", "amount": 150, "description": "$150 at Trader Joe's", "merchant": "Trader Joe's"}
    simulated_agent10_bid = {"bid": 150 * 0.042, "recommended_card": "agent10-card"}
    sample_offers_text = "Sample competitor offers text here."
    result = asyncio.run(judge.process_transaction(transaction, simulated_agent10_bid, sample_offers_text))
    print(json.dumps(result, indent=2))
