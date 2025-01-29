import json
from typing import Dict
from openai import AsyncOpenAI


class Agent4_external_events_AI:
    def __init__(self):
        self.client = AsyncOpenAI(api_key="sk-g8hvV0zoMOD29zq0zhV9n4MIwAmSoh65iJgEybbpIeT3BlbkFJ3YTkPDnHR-hzrrZzLdIy7H6-dKcP3I1YYbnJisnqkA")


    def load_trends_from_json(self, file_path: str) -> dict:
        """
        Load trends data from a JSON file and optimize it.
        """

        try:
            with open(file_path, "r") as f:
                trends_data = json.load(f)
                print("Loaded trends data from file.")

            return trends_data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return {}

    async def generate_analysis(self, events_file: str, agent3_file: str,company_name: str, question: str) -> Dict:
        """
        Generate performance analysis based on trends data and the user-provided question.
        """
        try:
            # Load trends data from JSON file
            trends_data = self.load_trends_from_json(agent3_file)
            if not trends_data:
                return {"error": "Failed to load trends data from JSON file."}
            event_data = self.load_trends_from_json(events_file)
            if not event_data:
                return {"error": "Failed to load event data from JSON file."}
            # Prepare the system prompt
            system_prompt = (
                "You are an expert in consumer behavior, credit card business, marketing, risk analysis. "
                "your training data includes trend summary data and world events data."
                "Your goal is to analyze impacts of major world or demestic events such as COVID 19, inflation included in the file to explain the google search trends of credit card related products or business activities. "
                "Focus on explaining the trends, highlighting potential causes, and do not over react to the impact of some events. "
                "For each world business or social event, try to explain the correlation with the trends and explain why the trend is increasing or decreasing."
                "Consider some events might have immediate impact, some might have long term impacts."
                "Try to Structure your response as follows: "
                "\n1. Summary of events (summary of event impact to the trend) "
                "\n2. Key Insights (what are the key insights you can find between the events and the trends, any strong correlation?)"
                "\n3. Recommendations (which event has the strongest correlation with the trend, any new opportunities)"
                "\n4. Key events and trends with Timeframes (list specific events and correlated trends with exact timeframes, events descriptions, and trend percentage changes, structured like this:"
                '\n- {"timeframe": "[Jan 2020 - Sep 2021]", "description": "Venture card Google trend search decreased by 40% and remain low, which could casue by the COVID 19 event"}).'
            )

            # Format the trends data as input for ChatGPT
            trends_data_str = json.dumps(trends_data, indent=2)
            events_data_str = json.dumps(event_data, indent=2)
            # Create the user message with trends data
            user_message = (
                f"Company Name: {company_name}\n"
                f"Question: {question}\n\n"
                f"Trends Data:\n{trends_data_str}\n\n"
                f"Event Data:\n{events_data_str}\n\n"
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
            content = response.choices[0].message.content.strip()

            formatted_response = {
                "summary": None,
                "insights": None,
                "recommendations": None,
                "Key_events_trends_Timeframes": []
            }

            # Use regex for safer extraction
            summary_match = re.search(r"\n1\. Summary of Events Impact to the Trends:\n(.*?)\n2\. Key Insights:\n", content, re.DOTALL)
            if summary_match:
                formatted_response["summary"] = summary_match.group(1).strip()

            insights_match = re.search(r"\n2\. Key Insights:\n(.*?)\n3\. Recommendations:\n", content, re.DOTALL)
            if insights_match:
                formatted_response["insights"] = insights_match.group(1).strip()

            recommendations_match = re.search(r"\n3\. Recommendations:\n(.*?)\n4\. Key events and trends with Timeframes:", content, re.DOTALL)
            if recommendations_match:
                formatted_response["recommendations"] = recommendations_match.group(1).strip()

            if "\n4. Key events and trends with Timeframes:" in content:
                trends_raw = content.split("\n4. Key events and trends with Timeframes:")[1].strip()
                for line in trends_raw.split("\n-"):
                    if line.strip():
                        trend = line.strip().lstrip("-").strip()
                        try:
                            formatted_response["Key_events_trends_Timeframes"].append(json.loads(trend))
                        except json.JSONDecodeError:
                            formatted_response["Key_events_trends_Timeframes"].append({"description": trend})

            return formatted_response


        except Exception as e:
            print(f"Error generating performance analysis: {e}")
            return {"error": str(e)}





