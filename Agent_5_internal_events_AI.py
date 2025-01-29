import json
from typing import Dict
from openai import AsyncOpenAI


class Agent5_internal_events_AI:
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

    async def generate_analysis(self, I_events_file: str, agent3_file: str,company_name: str, question: str) -> Dict:
        """
        Generate performance analysis based on trends data and the user-provided question.
        """
        try:
            # Load trends data from JSON file
            trends_data = self.load_trends_from_json(agent3_file)
            if not trends_data:
                return {"error": "Failed to load trends data from JSON file."}
            event_data = self.load_trends_from_json(I_events_file)
            if not event_data:
                return {"error": "Failed to load event data from JSON file."}
            # Prepare the system prompt
            system_prompt = (
                "You are an expert in consumer behavior, credit card business, marketing, risk analysis. "
                "Your goal is to analyze impacts of company promotion or news events such as 0% APR promotion, customer lawsuit included in the file to explain the google search trends of credit card related products or business activities. "
                "Focus on explaining the trends, highlighting potential causes, and do not over react to the impact of some events. "
                "For each event, try to explain the correlation with the trends and explain why the trend is increasing or decreasing. "
                "Consider some events might have immediate impact, some might have long term impacts."
                "Try to Structure your response as follows: "
                "\n1. Summary of events impact to the trends "
                "\n2. Key Insights "
                "\n3. Recommendations"
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

            # Extract the response content
            content = response.choices[0].message.content.strip()
            formatted_response = {
                "summary": None,
                "insights": None,
                "recommendations": None,
                "raw_content": content.strip()  # Always include raw content as fallback
            }

            if "\n1. Summary of Trends\n" in content:
                formatted_response["summary"] = content.split("\n1. Summary of Trends\n")[1].split("\n2. Key Insights\n")[0].strip() if "\n2. Key Insights\n" in content else content.split("\n1. Summary of Trends\n")[1].strip()

            if "\n2. Key Insights\n" in content:
                formatted_response["insights"] = content.split("\n2. Key Insights\n")[1].split("\n3. Recommendations\n")[0].strip() if "\n3. Recommendations\n" in content else content.split("\n2. Key Insights\n")[1].strip()

            if "\n3. Recommendations\n" in content:
                formatted_response["recommendations"] = content.split("\n3. Recommendations\n")[1].strip()
            # Save analysis to JSON file
            output_path = "trend_data_files/agent_5_output.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as file:
                json.dump(formatted_response, file, indent=2)
            return formatted_response

        except Exception as e:
            print(f"Error generating performance analysis: {e}")
            return {"error": str(e)}










