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

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)

#OPENAI_API_KEY = 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


class MarketTrendsAgent:
    def __init__(self):
        # Initialize pytrends API
        self.pytrends = TrendReq(hl='en-US', tz=360)
        # Initialize OpenAI API client
        self.client = AsyncOpenAI(api_key="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")

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
        
    def save_trends_to_json(self, trends_data: list, output_file: str = "all_trends.json") -> None:
        """
        Save the trends data to a JSON file in the desired format.

        Args:
            trends_data (list): A list of trend dictionaries, where each dictionary contains trend data.
            output_file (str): The file path to save the JSON file.
        """
        try:
            formatted_data = {}

            for trend in trends_data:
                if "trend" in trend:
                    trend_info = trend["trend"]
                    search_term = trend_info["search_term"]
                    formatted_data[search_term] = {
                        "search_term": trend_info["search_term"],
                        "average_interest": trend_info["average_interest"],
                        "peak_interest": trend_info["peak_interest"],
                        "latest_interest": trend_info["latest_interest"],
                        "trend_data": {str(k): v for k, v in trend_info["trend_data"].items()},
                    }

            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save the formatted data to a JSON file
            with open(output_file, "w") as json_file:
                json.dump(formatted_data, json_file, indent=4)
            logging.info(f"Trend data saved successfully to {output_file}")

        except Exception as e:
            logging.error(f"Error saving trends to JSON file: {e}")
        

    def get_trend_data(self, keyword: str, company: str, timeframe: str = "today 12-m", min_data_ratio: float = 0.7) -> dict:
        """
        Fetch Google Trends data for a single keyword, incorporating the company name, with threshold logic.
        
        Args:
            keyword (str): The search keyword.
            company (str): The company name for context.
            timeframe (str): Timeframe for the trend data (default: 'today 12-m').
            min_data_ratio (float): Minimum ratio of non-zero data points required for valid trend data.
            
        Returns:
            dict: Trend data if valid; otherwise, a dictionary with an error message.
        """
        try:
            search_term = f"{company} {keyword}"
            self.pytrends.build_payload([search_term], cat=0, timeframe=timeframe, geo="US", gprop="")
            data = self.pytrends.interest_over_time()

            if data.empty:
                logging.warning(f"No data available for search term: {search_term}")
                return {"error": f"No trend data available for search term: {search_term}"}

            trend_data = data[search_term].dropna()
            total_points = len(trend_data)
            valid_points = sum(1 for value in trend_data if value > 0)
            data_ratio = valid_points / total_points if total_points > 0 else 0

            if data_ratio < min_data_ratio:
                logging.warning(
                    f"Data quality issue for search term: {search_term}. "
                    f"Valid ratio: {data_ratio:.2f} (below threshold: {min_data_ratio})."
                )
                return {"error": f"Data quality below threshold for search term: {search_term}"}

            trend_summary = {
                "search_term": search_term,
                "average_interest": float(trend_data.mean()),
                "peak_interest": int(trend_data.max()),
                "latest_interest": int(trend_data.iloc[-1]),
                "trend_data": trend_data.to_dict(),
            }

            # Log a summary of the data
            logging.info(
                f"Trend Data Summary for '{search_term}': "
                f"Average: {trend_summary['average_interest']}, "
                f"Peak: {trend_summary['peak_interest']}, "
                f"Latest: {trend_summary['latest_interest']}"
            )

            # Optionally log the full data at the DEBUG level
            logging.debug(f"Full Trend Data for '{search_term}': {trend_summary}")

            return {"trend": trend_summary}
        except Exception as e:
            logging.error(f"Error fetching trend data for '{keyword}' with company '{company}': {str(e)}")
            return {"error": f"Failed to fetch trend data for keyword '{keyword}' with company '{company}': {str(e)}"}
        
        finally:
            time.sleep(5) 

    def generate_trend_graph(self, trends: dict, title: str = "Google Trends Analysis") -> str:
        """
        Generate a graph for trend data and return it as a base64-encoded string.
        
        Args:
            trends (dict): A dictionary containing validated trend data.
            title (str): Title for the graph (default: 'Google Trends Analysis').
            
        Returns:
            str: A base64-encoded string of the graph or an empty string if no valid data exists.
        """
        try:
            if not trends:
                print("No trend data available to plot.")
                return ""

            plt.figure(figsize=(12, 8))
            line_styles = ['-', '--', '-.', ':']
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            style_cycle = [(color, style) for style in line_styles for color in colors]

            valid_data_found = False  # Track if any valid data exists for graphing

            # Plot each trend
            for idx, (keyword, data) in enumerate(trends.items()):
                # Skip trends without valid data
                if "trend_data" not in data or not data["trend_data"]:
                    print(f"No valid trend data for keyword: {keyword}, skipping.")
                    continue

                trend_data = pd.Series(data["trend_data"])
                if trend_data.empty:
                    logging.warning(f"Trend data for keyword '{keyword}' is empty. Skipping.")
                    continue                
                color, line_style = style_cycle[idx % len(style_cycle)]
                trend_data.plot(label=keyword, linestyle=line_style, color=color)
                valid_data_found = True  # Mark that valid data exists

            if not valid_data_found:
                logging.warning("No valid trend data available to generate graph.")
                return ""

            plt.title(title, fontsize=16)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Interest", fontsize=12)
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.legend(loc='best', fontsize=10)
            plt.tight_layout()

            # Save the graph to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150)
            buffer.seek(0)
            plt.close()

            graph_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            buffer.close()
            # Add the required prefix to the Base64 string
            graph_base64 = f"data:image/png;base64,{graph_base64}"

            logging.info(f"Generated graph successfully. Base64 size: {len(graph_base64)}")
            logging.debug(f"Graph Base64: {graph_base64[:50]}... (truncated)")
            return graph_base64
        except Exception as e:
            logging.error(f"Error generating trend graph: {str(e)}")
            return ""


    async def analyze_trends(self, company: str, question: str, timeframe: str = "today 12-m") -> dict:
        """
        Analyze trends and generate graphs for each keyword.
        """
        try:
            keywords = await self.generate_keywords(question)
            if not keywords:
                return {"error": "Failed to generate keywords for analysis."}

            trends_data = []
            graphs = {}
            failed_keywords = []

            for keyword in keywords:
                logging.info(f"Processing keyword: {keyword}")
                trend = self.get_trend_data(keyword, company, timeframe)
                if not trend:
                    failed_keywords.append(keyword)
                    continue

                trends_data.append(trend)
                graph = self.generate_trend_graph(trend, title=f"Trends for '{keyword}'")
                graphs[keyword] = graph if graph else None

            # Save trends to JSON file
            self.save_trends_to_json(trends_data, output_file="trend_data_files/all_trends.json")

            return {
                "trends": trends_data,
                "graphs": graphs,
                "failed_keywords": failed_keywords,
            }
        except Exception as e:
            logging.error(f"Error analyzing trends: {e}")
            return {"error": f"Failed to analyze trends: {e}"}