from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from openai import AsyncOpenAI
import json
import logging
import os
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)


class MarketTrendsAgent:
    def __init__(self):
        # Initialize pytrends API
        self.pytrends = TrendReq(hl='en-US', tz=360)
        # Initialize OpenAI API client
        self.client = AsyncOpenAI(api_key="sk-g8hvV0zoMOD29zq0zhV9n4MIwAmSoh65iJgEybbpIeT3BlbkFJ3YTkPDnHR-hzrrZzLdIy7H6-dKcP3I1YYbnJisnqkA")

    async def generate_keywords(self, question: str) -> list:
        """
        Generate keywords dynamically from a question using OpenAI's GPT-4.
        """
        try:
            # Construct the prompt
            system_prompt = (
                "You are an expert in analyzing user questions and generating highly relevant, flexible keywords for search engines. Your task is to create a valid JSON array of concise keywords for Google Trends analysis. "
                "Output strictly as a JSON array of strings. Do not include explanations, comments, or additional text. Only the JSON array should be returned."
                "Add flexibility by including slight variations or related terms, but ensure they remain closely tied to the main topic or intent of the query."
                "Avoid overly broad or generic terms unless they naturally align with the query's context."
                "Strike a balance between precision and exploration to capture both direct and closely related trends."
                "If unsure, prioritize clarity and user intent over adding excessive variations."
            )
            user_message = (
                f"Question: {json.dumps(question)}\n\n"  # Escape question properly
                f"Generate keywords as a JSON array of strings:"
            )

            # Call the OpenAI API
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=150,
                temperature=0.5,
            )

            # Extract the generated keywords
            content = response.choices[0].message.content.strip()

            # Extract JSON-like text
            try:
                # Ensure content is a valid JSON array
                if not content.startswith("[") or not content.endswith("]"):
                    raise ValueError("The output is not a valid JSON array.")
                
                keywords = json.loads(content)
                # Limit to 4 keywords
                keywords = keywords[:5]
                #keywords =['Sapphire Preferred benefit','Sapphire Preferred','Sapphire Reserve','Sapphire Reserve benefit', 'Credit Card']
            except json.JSONDecodeError as json_err:
                print(f"Malformed JSON detected in AI response: {content}")
                raise json_err

            print(f"Generated keywords: {keywords}")
            return keywords

        except json.JSONDecodeError as json_err:
            print(f"JSON decode error: {str(json_err)}")
            return []

        except Exception as e:
            print(f"Error generating keywords with OpenAI: {str(e)}")
            return []
        

    def get_trend_data(
        self, keywords: list, company: str, timeframe: str = "today 12-m", min_data_ratio: float = 0.7
    ) -> None:
        """
        Fetch Google Trends data for multiple keywords, incorporating the company name, and save to two separate files:
        1. Detailed trend data (dates and values)
        2. Summary statistics (average, peak, latest interest, etc.)
        """
        try:
            detailed_trends_data = {}
            summary_trends_data = {}

            for keyword in keywords:
                search_term = f"{company} {keyword}"
                self.pytrends.build_payload([search_term], cat=0, timeframe=timeframe, geo="US", gprop="")
                data = self.pytrends.interest_over_time()

                if data.empty:
                    logging.warning(f"No data available for search term: {search_term}")
                    continue

                trend_data = data[search_term].dropna()
                total_points = len(trend_data)
                valid_points = sum(1 for value in trend_data if value > 0)
                data_ratio = valid_points / total_points if total_points > 0 else 0

                if data_ratio < min_data_ratio:
                    logging.warning(
                        f"Data quality issue for search term: {search_term}. "
                        f"Valid ratio: {data_ratio:.2f} (below threshold: {min_data_ratio})."
                    )
                    continue

                # Format trend data
                formatted_trend_data = {str(date)[:10]: value for date, value in trend_data.to_dict().items()}

                # Save to detailed trends dictionary
                detailed_trends_data[search_term] = {
                    "trend_data": formatted_trend_data,
                }

                # Save to summary trends dictionary
                summary_trends_data[search_term] = {
                    "search_term": search_term,
                    "average_interest": float(trend_data.mean()),
                    "peak_interest": int(trend_data.max()),
                    "latest_interest": int(trend_data.iloc[-1]),
                }

            # Save detailed trend data to a JSON file
            os.makedirs("trend_data_files", exist_ok=True)
            detailed_trends_file = "trend_data_files/all_trends.json"
            with open(detailed_trends_file, "w", encoding="utf-8") as file:
                json.dump(detailed_trends_data, file, indent=4, ensure_ascii=False)
            logging.info(f"Detailed trends data saved successfully to '{detailed_trends_file}'.")

            # Save summary trend data to a JSON file
            summary_trends_file = "trend_data_files/summary_trends.json"
            with open(summary_trends_file, "w", encoding="utf-8") as file:
                json.dump(summary_trends_data, file, indent=4, ensure_ascii=False)
            logging.info(f"Summary trends data saved successfully to '{summary_trends_file}'.")

        except Exception as e:
            logging.error(f"Error fetching trend data: {e}")

        finally:
            # Sleep for a random time to avoid rate limiting
            sleep_duration = random.uniform(5, 15)  # Random sleep between 5 and 15 seconds
            logging.info(f"Sleeping for {sleep_duration:.2f} seconds to avoid rate limiting.")
            time.sleep(sleep_duration)


    async def analyze_trends(self, company: str, question: str, timeframe: str = "today 12-m") -> dict:
        """
        Analyze trends and return data for WebSocket communication.
        """
        try:
            # Generate keywords
            keywords = await self.generate_keywords(question)
            if not keywords:
                return {"error": "Failed to generate keywords for analysis."}

            # Fetch and save trend data
            self.get_trend_data(keywords, company, timeframe)

            # Load trend data from the JSON file
            detailed_trends_file = "trend_data_files/all_trends.json"
            try:
                with open(detailed_trends_file, "r", encoding="utf-8") as file:
                    trends_data = json.load(file)
            except FileNotFoundError:
                return {"error": f"Trend data file not found: {detailed_trends_file}"}
            except json.JSONDecodeError as e:
                return {"error": f"Error decoding trend data file: {str(e)}"}

            # Return trends data directly along with the file paths
            return {
                "trends_data": trends_data,  # Include trends directly
                "detailed_trends_file": detailed_trends_file,
                "summary_trends_file": "trend_data_files/summary_trends.json",
            }

        except Exception as e:
            logging.error(f"Error analyzing trends: {e}")
            return {"error": f"Failed to analyze trends: {e}"}
