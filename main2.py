import asyncio
import json
import logging
from Agent_10_Angie import AIAgent10
from Agent_9_1_competitor_offer import OfferWinerAgent
from Agent_9_5_offerwiner_judge import OfferWinerAgentJudge

logging.basicConfig(level=logging.INFO)

async def run_agent10():
    OPENAI_API_KEY = ""
    # Instantiate Agent 10 with an initial budget.
    agent10 = AIAgent10(OPENAI_API_KEY, initial_budget=20.0)
    # Example transactions from the front end.
    transactions = [
        {"category": "groceries", "amount": 150, "description": "$150 at Trader Joe's", "merchant": "Trader Joe's"},
        {"category": "groceries", "amount": 200, "description": "$200 at Costco", "merchant": "Costco"},
        {"category": "groceries", "amount": 50,  "description": "$50 at Mariano's", "merchant": "Mariano's"},
        {"category": "restaurant", "amount": 75, "description": "$75 dinner at a local restaurant", "merchant": "Local Restaurant"},
        {"category": "gas", "amount": 100, "description": "$100 fuel purchase", "merchant": "Shell"},
        {"category": "uber", "amount": 80, "description": "$80 spent on Uber rides", "merchant": "Uber"},
        {"category": "travel", "amount": 250, "description": "$250 airline ticket", "merchant": "United Airlines"},
        {"category": "utilities", "amount": 150, "description": "$150 utility bills", "merchant": "ComEd"},
        {"category": "entertainment", "amount": 120, "description": "$120 for movies/events", "merchant": "AMC Theatres"},
        {"category": "online shopping", "amount": 200, "description": "$200 on online shopping", "merchant": "Amazon"},
        {"category": "cellphone", "amount": 80, "description": "$80 cellphone bill", "merchant": "Verizon"},
        {"category": "health", "amount": 100, "description": "$100 on healthcare/medications", "merchant": "CVS Pharmacy"}
    ]
    # Competitor card keys provided from the front end.
    competitor_keys = ["discover-cashback", "chase-freedom", "capitalone-quicksilver", "wellsfargo-cashwise"]
    ai_bid_results = await agent10.process_transactions(transactions, competitor_keys)
    return transactions, competitor_keys, ai_bid_results

def run_offer_winer_agent_offers_text(competitor_keys):
    """
    Synchronously prepare competitor offers text using OfferWinerAgent.
    """
    OPENAI_API_KEY = "your-openai-api-key-here"
    agent = OfferWinerAgent(OPENAI_API_KEY)
    offers_text = agent.get_competitor_offers_text(competitor_keys)
    return offers_text

async def run_judge_agent(transactions, ai_bid_results, competitor_offers_text):
    OPENAI_API_KEY = "your-openai-api-key-here"
    judge_agent = OfferWinerAgentJudge(OPENAI_API_KEY)
    judge_results = await judge_agent.process_transactions(transactions, ai_bid_results, competitor_offers_text)
    return judge_results

async def main():
    # Run Agent 10 to generate bid information.
    transactions, competitor_keys, ai_bid_results = await run_agent10()
    
    # Obtain competitor offers text from OfferWinerAgent.
    competitor_offers_text = run_offer_winer_agent_offers_text(competitor_keys)
    
    # Run Judge Agent to compare Agent 10’s bid with competitor offers.
    judge_results = await run_judge_agent(transactions, ai_bid_results, competitor_offers_text)
    
    # Display combined results.
    for i, tx in enumerate(transactions):
        print("Transaction:", json.dumps(tx, indent=2))
        print("Agent 10 Bid Result:", json.dumps(ai_bid_results[i], indent=2))
        print("Competitor Offers Text:\n", competitor_offers_text)
        print("Judge Decision:", json.dumps(judge_results[i], indent=2))
        print("-" * 40)
    
if __name__ == "__main__":
    asyncio.run(main())
