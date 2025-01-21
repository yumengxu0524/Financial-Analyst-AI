import json
import difflib
from typing import Dict, List, Optional
from openai import AsyncOpenAI
import requests


json_file_path = "C:/Users/ymx19/DISCOVER/financial_variables.json"

class Agent1:
    def __init__(self, json_file_path: str, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.base_url = "https://www.alphavantage.co/query"  # Base URL for Alpha Vantage API
        
        # Load financial variables from JSON
        with open(json_file_path, 'r') as file:
            self.financial_variables = json.load(file)

    async def generate_keywords(self, question: str) -> List[str]:
        """
        Generate keywords dynamically from a question using OpenAI's GPT-4.
        """
        try:
            system_prompt = (
                "You are an expert in financial data analysis. Extract the key financial metrics from the user's question. "
                "Return them as a JSON array of concise keywords. Output only the JSON array without additional text."
            )
            user_message = f"Question: {json.dumps(question)}\n\nExtract keywords:"
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=150,
                temperature=0.5,
            )
            
            # Parse the JSON array of keywords
            content = response.choices[0].message.content.strip()
            if not content.startswith("[") or not content.endswith("]"):
                raise ValueError("The output is not a valid JSON array.")
            
            keywords = json.loads(content)
            return keywords
        except Exception as e:
            print(f"Error generating keywords: {e}")
            return []

    def match_keywords(self, keywords: List[str]) -> List[str]:
        """
        Match keywords to financial variables using fuzzy matching.
        """
        matches = []
        for keyword in keywords:
            match = difflib.get_close_matches(keyword.lower(), self.financial_variables.keys(), n=1, cutoff=0.6)
            if match:
                matches.append(self.financial_variables[match[0]])
        return matches

    def fetch_financial_data(self, symbol: str, variables: List[str], annual: bool = False) -> Optional[Dict[str, List[Dict]]]:
        """
        Fetch financial data for all sheets (income statement, balance sheet, cash flow) from Alpha Vantage API.
        """
        try:
            function_map = {
                "income_statement": "INCOME_STATEMENT",
                "balance_sheet": "BALANCE_SHEET",
                "cash_flow": "CASH_FLOW",
            }

            all_data = {}

            for sheet_name, api_function in function_map.items():
                params = {
                    "function": api_function,
                    "symbol": symbol,
                    "apikey": "MUW1G1BPCMPUOLWJ",
                }
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                # Log the API response for debugging
                print(f"Alpha Vantage API response for {sheet_name}: {json.dumps(data, indent=2)}")

                # Extract and filter relevant data
                report_type = "annualReports" if annual else "quarterlyReports"
                if report_type in data:
                    filtered_data = [
                        {var: report.get(var.lower(), "Data not available") for var in variables}
                        for report in data[report_type]
                    ]
                    all_data[sheet_name] = filtered_data
                else:
                    print(f"No {report_type} found in the response for {sheet_name}.")
                    all_data[sheet_name] = []

            # Log combined data for debugging
            print(f"Combined financial data: {json.dumps(all_data, indent=2)}")
            return all_data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None


    async def process_request(self, question: str, time_range: str) -> Dict:
        # Step 1: Generate keywords
        keywords = await self.generate_keywords(question)
        print(f"Generated keywords: {keywords}")

        # Step 2: Match keywords to financial variables
        matched_variables = self.match_keywords(keywords)
        print(f"Matched variables: {matched_variables}")

        # Step 3: Fetch financial data
        # Replace "AAPL" with the actual company symbol passed from the frontend
        financial_data = self.fetch_financial_data("AAPL", matched_variables, annual=True)

        # Ensure the response always includes financial_data
        result = {
            "variables": matched_variables,
            "time_range": time_range,
            "financial_data": financial_data if financial_data else []
        }

        # Log the result for debugging
        print(f"Returning result: {json.dumps(result, indent=2)}")
        return result

