import json
from typing import Dict
from openai import AsyncOpenAI

    
class Agent4_marketing_AI:
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)


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

        try:
            with open(file_path, "r") as f:
                trends_data = json.load(f)
                print("Loaded trends data from file.")

            # Apply optimizations
            trends_data = remove_time_from_timestamps(trends_data)
            trends_data = reduce_decimal_precision(trends_data)

            return trends_data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
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

            # Prepare the system prompt
            system_prompt = (
                "You are an expert in finance, credit card business, marketing, risk analysis, and data analysis. "
                "Your goal is to analyze Google Trends data provided for a specific company and provide performance insights. "
                "Focus on explaining the trends, highlighting potential causes, and making actionable recommendations. "
                "Try to Structure your response as follows: "
                "\n1. Summary of Trends "
                "\n2. Key Insights "
                "\n3. Recommendations"
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
            # Check for structured sections in the content
            formatted_response = {}
            if "\n1. Summary of Trends\n" in content:
                formatted_response["summary"] = content.split("\n1. Summary of Trends\n")[1].split("\n2. Key Insights\n")[0].strip() if "\n2. Key Insights\n" in content else content.split("\n1. Summary of Trends\n")[1].strip()
            if "\n2. Key Insights\n" in content:
                formatted_response["insights"] = content.split("\n2. Key Insights\n")[1].split("\n3. Recommendations\n")[0].strip() if "\n3. Recommendations\n" in content else content.split("\n2. Key Insights\n")[1].strip()
            if "\n3. Recommendations\n" in content:
                formatted_response["recommendations"] = content.split("\n3. Recommendations\n")[1].strip()

            # Fallback for raw content if no sections are found
            if not formatted_response:
                formatted_response = {
                    "summary": "Unable to generate a summary. The trends data might be insufficient or irrelevant.",
                    "insights": "Unable to provide insights due to limited or missing trends data.",
                    "recommendations": "Unable to provide recommendations based on the available data.",
                }

            return formatted_response

        except Exception as e:
            print(f"Error generating performance analysis: {e}")
            return {"error": str(e)}


