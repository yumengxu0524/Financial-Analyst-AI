from fastapi import FastAPI,  WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Agent_1_data_retrival import Agent1
from Agent_2_market_trends import MarketTrendsAgent
from Agent_3_performance_analysis import Agent3PerformanceAnalysis
from Agent_4_marketing_AI import Agent4_marketing_AI
import json

from dotenv import load_dotenv
load_dotenv()

FINANCIAL_VARIABLES_JSON = "/Users/xuyumeng/Desktop/Discover/Financial-Analyst-AI/financial_variables.json"


# Initialize agents with necessary configurations
OPENAI_API_KEY = "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"
ALPHAVANTAGE_API_KEY  = "MUW1G1BPCMPUOLWJ"
BASE_URL = "https://www.alphavantage.co/query"

agent1 = Agent1(
    json_file_path=FINANCIAL_VARIABLES_JSON,
    openai_api_key=OPENAI_API_KEY,
)


market_trends_agent = MarketTrendsAgent()

app = FastAPI()

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent 3
agent3 = Agent3PerformanceAnalysis(openai_api_key=OPENAI_API_KEY)
agent4 = Agent4_marketing_AI(openai_api_key=OPENAI_API_KEY)

@app.websocket("/ws/agent1")
async def websocket_agent1(websocket: WebSocket):
    """
    WebSocket endpoint for Agent1.
    Handles requests for financial variable matching and time range processing.
    """
    await websocket.accept()
    try:
        while True:
            try:
                # Receive data from the frontend
                data = await websocket.receive_text()
                request = json.loads(data)

                # Extract parameters from the request
                question = request.get("question", "").strip()
                time_range = request.get("time_range", "").strip()

                # Validate input
                if not question or not time_range:
                    await websocket.send_text(json.dumps({
                        "error": "Both question and time range are required."
                    }))
                    continue

                # Log received request (useful for debugging)
                print(f"Received request: question={question}, time_range={time_range}")

                # Call Agent1 to process the question
                response = await agent1.process_request(question, time_range)

                # Check and send the response
                if response.get("variables"):
                    await websocket.send_text(json.dumps(response))
                else:
                    await websocket.send_text(json.dumps({
                        "error": "No variables matched for the given question."
                    }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format in the request."
                }))
            except Exception as e:
                print(f"Unexpected error: {e}")
                await websocket.send_text(json.dumps({
                    "error": f"An unexpected error occurred: {str(e)}"
                }))
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    finally:
        await websocket.close()
        print("WebSocket connection closed.")


class AnalysisRequest(BaseModel):
    question: str
    time_range: str


@app.get("/")
async def root():
    """
    Root endpoint for testing.
    """
    return {"message": "Financial Analysis API is running!"}


def serialize_trend_data(trend_data):
    """
    Convert trend data keys (Timestamps) to strings for JSON serialization.
    """
    if isinstance(trend_data, dict):
        return {key.strftime('%Y-%m-%d'): value for key, value in trend_data.items()}
    return trend_data  # If it's not a dict, return as is to avoid errors.

def validate_trend_data(trend_data):
    """
    Validate trend data to ensure correct structure.
    """
    if not isinstance(trend_data, dict):
        print(f"Invalid trend data format: {type(trend_data)}. Expected a dictionary.")  # Debug log
        return trend_data  # Bypass validation if not a dictionary.

    valid_trend_data = {}
    for key, value in trend_data.items():
        try:
            # Ensure the trend data values are valid
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid data type for value: {value}")
            valid_trend_data[key] = value
        except ValueError as e:
            print(f"Skipping invalid trend key '{key}': {e}")  # Log invalid data
    return valid_trend_data


@app.websocket("/ws/agent2")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                request = json.loads(data)
                company = request.get("company", "").strip()
                question = request.get("question", "").strip()
                compare_companies = request.get("compare_companies", [])
                timeframe = request.get("timeframe", "today 12-m")

                if not question:
                    await websocket.send_text(json.dumps({"error": "A question is required."}))
                    continue

                if not company and not compare_companies:
                    await websocket.send_text(json.dumps({"error": "Provide a company or companies for analysis."}))
                    continue

                trends_to_compare = {}
                individual_graphs = {}

                # Analyze the primary company
                if company:
                    primary_trends = await market_trends_agent.analyze_trends(company, question, timeframe)
                    trend_data = primary_trends.get("trends", {})
                    validated_trend_data = validate_trend_data(trend_data)
                    if validated_trend_data:
                        trends_to_compare[company] = validated_trend_data
                        individual_graphs.update(primary_trends.get("graphs", {}))

                if trends_to_compare:
                    response = {
                        "text_output": f"Market trends for {company} analyzed successfully.",
                        "individual_graphs": individual_graphs
                    }
                    print("Response sent to frontend:", response)  # Debugging
                    await websocket.send_text(json.dumps(response))
                else:
                    await websocket.send_text(json.dumps({
                        "error": "No valid trend data available for the provided companies."
                    }))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"Failed to process request: {str(e)}"}))
    except WebSocketDisconnect:
        print("WebSocket disconnected.")




class AnalysisRequest(BaseModel):
    company: str
    quarter: str
    question: str = None  # Optional user question


@app.websocket("/ws/agent3")
async def websocket_agent3(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"[Agent 3] Received data: {data}")  # Log received data
            
            try:
                request = json.loads(data)
                company_name = request.get("company", "").strip()
                question = request.get("question", "").strip()

                if not company_name:
                    error_message = "Company name is required."
                    print(f"[Agent 3] Error: {error_message}")
                    await websocket.send_text(json.dumps({"error": error_message}))
                    continue

                # Provide the file path for trends data to Agent 3
                trends_file = "trend_data_files/all_trends.json"

                # Perform analysis
                analysis_result = await agent3.generate_analysis(trends_file, company_name, question)
                print(f"[Agent 3] Analysis Result: {analysis_result}")  # Log analysis result

                # Send response
                await websocket.send_text(json.dumps(analysis_result))

            except json.JSONDecodeError as e:
                error_message = f"JSON decode error: {e}"
                print(f"[Agent 3] Error: {error_message}")
                await websocket.send_text(json.dumps({"error": error_message}))
            except Exception as e:
                error_message = f"Unexpected error: {e}"
                print(f"[Agent 3] Error: {error_message}")
                await websocket.send_text(json.dumps({"error": error_message}))

    except WebSocketDisconnect:
        print("[Agent 3] WebSocket disconnected.")



@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Analysis Chat</title>
    </head>
    <body>
        <h1>Financial Analysis Chat</h1>
        <div id="messages" style="border:1px solid #ccc; width:50%; height:300px; overflow:auto; padding:10px;"></div>
        <input type="text" id="company" placeholder="Enter company name..." style="width:50%;" />
        <input type="text" id="question" placeholder="Ask a question..." style="width:50%;" />
        <button onclick="sendMessage()">Send</button>
        <script>
            var ws = new WebSocket("ws://127.0.0.1:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById("messages");
                try {
                    var data = JSON.parse(event.data);
                    var content = "<div>";
                    if (data.text_output) content += `<p>${data.text_output}</p>`;
                    if (data.graph) {
                        content += `<img src="${data.graph}" alt="Graph" style="width:100%; max-width:800px;" />`;
                    } else {
                        content += "<p style='color: red;'>No graph data received.</p>";
                    }
                    content += "</div>";
                    messages.innerHTML += content;
                    messages.scrollTop = messages.scrollHeight;
                } catch (e) {
                    messages.innerHTML += `<p style='color: red;'>Error processing WebSocket data: ${e.message}</p>`;
                }
            };
            function sendMessage() {
                var company = document.getElementById("company").value.trim();
                var question = document.getElementById("question").value.trim();
                if (!company || !question) {
                    alert("Please enter both a company name and a question.");
                    return;
                }
                ws.send(JSON.stringify({ company, question }));
            }
        </script>
    </body>
    </html>
    """