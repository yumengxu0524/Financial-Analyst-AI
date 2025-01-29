import json
from typing import Dict, Any
from openai import AsyncOpenAI
import logging
import tiktoken
import os
    
class Agent3PerformanceAnalysis:
    def __init__(self):
        self.client = AsyncOpenAI(api_key="sk-g8hvV0zoMOD29zq0zhV9n4MIwAmSoh65iJgEybbpIeT3BlbkFJ3YTkPDnHR-hzrrZzLdIy7H6-dKcP3I1YYbnJisnqkA")


    def load_trends_from_json(self, file_path: str) -> dict:
        """
        Load trends data from a JSON file and optimize it.
        """
        def remove_time_from_timestamps(data):
            """
            Remove unnecessary time (00:00:00) from the timestamps in trend_data.
            """
            for key, value in data.items():
                if "trend_data" in value:
                    value["trend_data"] = {
                        timestamp.split(" ")[0]: score
                        for timestamp, score in value["trend_data"].items()
                    }
            return data

        def reduce_decimal_precision(data):
            """
            Reduce decimal precision for numerical fields.
            """
            for key, value in data.items():
                if isinstance(value, dict):
                    for field in ["average_interest", "peak_interest", "latest_interest"]:
                        if field in value:
                            value[field] = round(value[field], 2)
            return data
        
        def filter_trend_data(data):
            """
            Select two data points per month from trend_data in order to save computing space.
            """
            for key, value in data.items():
                if "trend_data" in value:
                    trend_data = value["trend_data"]
                    filtered_data = {}
                    month_groups = {}

                    # Group data points by month
                    for date, score in trend_data.items():
                        month = date[:7]  # Extract "YYYY-MM" from "YYYY-MM-DD"
                        if month not in month_groups:
                            month_groups[month] = []
                        month_groups[month].append((date, score))

                    # Select the first and last data points of each month
                    for month, entries in month_groups.items():
                        entries.sort()  # Ensure entries are sorted by date
                        if len(entries) >= 2:
                            filtered_data[entries[0][0]] = entries[0][1]  # First data point
                            filtered_data[entries[-1][0]] = entries[-1][1]  # Last data point
                        elif entries:  # If only one entry is available
                            filtered_data[entries[0][0]] = entries[0][1]

                    value["trend_data"] = filtered_data
            return data
        
        def reduce_to_one_data_point_per_month(data, key):
            """
            Reduce trend_data for a specific entry to one data point per month.
            """
            if "trend_data" in data[key]:
                trend_data = data[key]["trend_data"]
                filtered_data = {}
                month_groups = {}

                # Group data points by month
                for date, score in trend_data.items():
                    month = date[:7]  # Extract "YYYY-MM"
                    if month not in month_groups:
                        month_groups[month] = []
                    month_groups[month].append((date, score))

                # Keep only the first data point of each month
                for month, entries in month_groups.items():
                    entries.sort()  # Ensure entries are sorted by date
                    filtered_data[entries[0][0]] = entries[0][1]

                data[key]["trend_data"] = filtered_data        
        
        def check_token_count(data: Dict[str, Any], max_tokens: int = 6000) -> bool:
            """
            Check the number of tokens in trends_data to ensure it does not exceed the limit.
            
            :param data: The trends data dictionary.
            :param max_tokens: Maximum allowed token count.
            :return: True if token count is within limits, False otherwise.
            """
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                total_tokens = sum(
                    len(encoding.encode(json.dumps(value)))
                    for value in data.values()
                )
                logging.info(f"Total token count: {total_tokens}")
                return total_tokens <= max_tokens
            except Exception as e:
                logging.error(f"Error calculating token count: {e}")
                return False

        def iteratively_reduce_data(data):
            """
            Iteratively reduce trend_data points and remove entries until token count is within the limit.
            """
            keys = list(data.keys())  # Get all keys

            for key in reversed(keys):  # Start from the last entry
                logging.info(f"Reducing data for entry: {key}")
                reduce_to_one_data_point_per_month(data, key)
                if check_token_count(data):
                    return data

            # If still over the limit, start deleting entire entries
            while keys and not check_token_count(data):
                key_to_remove = keys.pop()  # Remove the last entry
                logging.warning(f"Removing entry: {key_to_remove} to reduce token count.")
                del data[key_to_remove]

            return data

        try:
            with open(file_path, "r") as f:
                trends_data = json.load(f)
                logging.info("Successfully loaded trends data from file.")

            # Apply optimizations
            trends_data = remove_time_from_timestamps(trends_data)
            logging.info("Removed time from timestamps in trend_data.")

            trends_data = reduce_decimal_precision(trends_data)
            logging.info("Reduced decimal precision in numerical fields.")

            trends_data = filter_trend_data(trends_data)
            logging.info("Filtered trend_data to retain minimal data points per month.")

            # Check token count and iteratively reduce data if necessary
            if not check_token_count(trends_data):
                logging.warning("Token count exceeds the allowed limit of 7000.")
                trends_data = iteratively_reduce_data(trends_data)

            logging.info("Token count is within the allowed limit after reduction.")
            return trends_data

        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")

        return {}

    async def generate_analysis(self, trends_file: str, company_name: str, question: str) -> Dict:
        """
        Generate performance analysis based on trends data and the user-provided question.
        """
        try:
            # Load trends data from JSON file
            trends_data = self.load_trends_from_json(trends_file)
            if not trends_data:
                return {"error": "Failed to load trends data from JSON file."}

            # Updated system prompt with time series, seasonality, and correlation focus
            system_prompt = (
                "You are an expert in finance, credit card business, marketing, risk analysis, and data analysis. "
                "Your task is to analyze time series seasonality, correlations between data entries, and identify key trends with specific timeframes. "
                "Provide actionable insights in the following structured format: "
                "\n1. Summary of Trends (a brief overview of the data and key takeaways)."
                "\n2. Seasonality and Correlations (explain seasonal patterns, correlations, and notable events, including their impact on consumer behavior)."
                "\n3. Recommendations (specific actions to take during high or low trends)."
                "\n4. Key Trends with Timeframes (list specific trends with exact timeframes, descriptions, and percentage changes, structured like this:"
            )

            

            # Format the trends data as input for ChatGPT
            trends_data_str = json.dumps(trends_data, indent=2)

            # Create the user message with trends data
            user_message = (
                f"Company Name: {company_name}\n"
                f"Question: {question}\n\n"
                f"Trends Data:\n{trends_data_str}\n\n"
                f"Provide a detailed analysis based on the above."
            )

            # Call the OpenAI API
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1000,
                temperature=0.7,
            )

            # Extract the response content
            content = response.choices[0].message.content.strip()
            
            # Parse response into structured sections
            formatted_response = {
                "summary": None,
                "seasonality_correlations": None,
                "recommendations": None,
                "key_trends_with_timeframes": None,
                "raw_content": content.strip()  # Pass entire content as fallback
            }

            # Parse structured sections if available
            if "\n1. Summary of Trends:" in content:
                formatted_response["summary"] = content.split("\n1. Summary of Trends:")[1].split("\n2. Seasonality and Correlations:")[0].strip()
            if "\n2. Seasonality and Correlations:" in content:
                formatted_response["seasonality_correlations"] = content.split("\n2. Seasonality and Correlations:")[1].split("\n3. Recommendations:")[0].strip()
            if "\n3. Recommendations:" in content:
                formatted_response["recommendations"] = content.split("\n3. Recommendations:")[1].split("\n4. Key Trends with Timeframes:")[0].strip()
            if "\n4. Key Trends with Timeframes:" in content:
                trends_raw = content.split("\n4. Key Trends with Timeframes:")[1].strip()
                formatted_response["key_trends_with_timeframes"] = []
                for line in trends_raw.split("\n-"):
                    if line.strip():
                        trend = line.strip().lstrip("-").strip()
                        try:
                            formatted_response["key_trends_with_timeframes"].append(eval(trend))  # Parse JSON-like trends
                        except Exception:
                            formatted_response["key_trends_with_timeframes"].append({"description": trend})


            # Save analysis to JSON file
            output_path = "trend_data_files/agent_3_output.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as file:
                json.dump(formatted_response, file, indent=2)

            return formatted_response

        except Exception as e:
            print(f"Error generating performance analysis: {e}")
            return {"error": str(e)}


