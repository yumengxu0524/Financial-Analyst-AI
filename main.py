from fastapi import FastAPI,  WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from Agent_1_data_retrival import Agent1
from Agent_2_market_trends import MarketTrendsAgent
from Agent_3_performance_analysis import Agent3PerformanceAnalysis
from Agent_4_external_events_AI import Agent4_external_events_AI
from Agent_5_internal_events_AI import Agent5_internal_events_AI
import json
from fastapi.staticfiles import StaticFiles
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


FINANCIAL_VARIABLES_JSON = "C:/Users/ymx19/DISCOVER/financial_variables.json"

# Initialize agents with necessary configurations
OPENAI_API_KEY = "sk-g8hvV0zoMOD29zq0zhV9n4MIwAmSoh65iJgEybbpIeT3BlbkFJ3YTkPDnHR-hzrrZzLdIy7H6-dKcP3I1YYbnJisnqkA"
ALPHAVANTAGE_API_KEY  = "MUW1G1BPCMPUOLWJ"
BASE_URL = "https://www.alphavantage.co/query"

agent1 = Agent1(
    json_file_path=FINANCIAL_VARIABLES_JSON,
    openai_api_key=OPENAI_API_KEY,
)

app = FastAPI()
app.mount("/trend_data_files", StaticFiles(directory="trend_data_files"), name="trend_data_files")

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
market_trends_agent = MarketTrendsAgent()

# Initialize Agent 3,4,5
agent3 = Agent3PerformanceAnalysis()
agent4 = Agent4_external_events_AI()
agent5 = Agent5_internal_events_AI()

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

@app.get("/trend_data_files/all_trends.json")
async def get_trend_file():
    """
    Endpoint to serve the all_trends.json file.
    """
    file_path = "trend_data_files/all_trends.json"
    return FileResponse(file_path)

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
async def websocket_agent2(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                request = json.loads(data)
                company = request.get("company", "").strip()
                question = request.get("question", "").strip()
                timeframe = request.get("timeframe", "today 12-m")

                if not question:
                    await websocket.send_text(json.dumps({"error": "A question is required."}))
                    continue
                # ---- STEP 1: Get trends data ------------------------------------------------
                # Analyze trends and fetch the data
                primary_trends = await market_trends_agent.analyze_trends(company, question, timeframe)

                if "error" in primary_trends:
                    await websocket.send_text(json.dumps({"error": primary_trends["error"]}))
                    continue
                # ---- STEP 2: Load events data from all_events.json -------------------------
                # You can load this file once per request (as shown) or once at startup
                # (caching it as a global variable if it's not changing).
                try:
                    with open("trend_data_files/all_events.json", "r") as f:
                        events_data = json.load(f)
                except FileNotFoundError:
                    events_data = {"error": "all_events.json not found."}
                except json.JSONDecodeError:
                    events_data = {"error": "Invalid JSON format in all_events.json."}
                # Send trends data directly in WebSocket response
                response = {
                    "text_output": f"Market trends for {company} analyzed successfully.",
                    "trends": primary_trends["trends_data"],  # Include trends directly
                    "eventsData": events_data   # The global events data
                }
                await websocket.send_text(json.dumps(response))
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

                # File paths for trends and events
                trends_file = "trend_data_files/all_trends.json"
                agent_3_file = "trend_data_files/agent_3_output.json"
                events_file = "trend_data_files/all_events.json"

                # Perform Agent 3 analysis
                agent3_result = await agent3.generate_analysis(trends_file, company_name, question)
                print(f"[Agent 3] Analysis Result: {agent3_result}")  # Log Agent 3 result

                # Save Agent 3 result to file if necessary
                with open(agent_3_file, "w") as f:
                    json.dump(agent3_result, f, indent=4)

                # If Agent 3 fails, send only its error
                if "error" in agent3_result:
                    await websocket.send_text(json.dumps(agent3_result))
                    continue

                # Perform Agent 4 analysis using the saved file
                agent4_result = await agent4.generate_analysis(events_file, agent_3_file, company_name, question)
                print(f"[Agent 4] Analysis Result: {agent4_result}")  # Log Agent 4 result

                # Combine Agent 3 and Agent 4 results
                combined_result = {
                    "agent3_analysis": agent3_result,
                    "agent4_analysis": agent4_result,
                }
                print(f"[Agent 3] Combined data sent successfully.")  # Log combined data

                # Send combined response
                await websocket.send_text(json.dumps(combined_result))

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



COMPANY_FILE_MAP = {
    "american express": "trend_data_files/company_data_file/American_Express.json",
    "amx": "trend_data_files/company_data_file/American_Express.json",
    "amex": "trend_data_files/company_data_file/American_Express.json",
    "bank of america": "trend_data_files/company_data_file/Bank_of_American.json",
    "boa": "trend_data_files/company_data_file/Bank_of_American.json",
    "capital one": "trend_data_files/company_data_file/Capital_One.json",
    "c1": "trend_data_files/company_data_file/Capital_One.json",
    "chase": "trend_data_files/company_data_file/Chase.json",
    "discover": "trend_data_files/company_data_file/Discover.json",
    "dfs": "trend_data_files/company_data_file/Discover.json",
    "wells fargo": "trend_data_files/company_data_file/Wells_Fargo.json",
    "wf": "trend_data_files/company_data_file/Wells_Fargo.json"
}

@app.websocket("/ws/agent5")
async def websocket_agent5(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"[Agent 5] Received data: {data}")  # Log received data
            
            try:
                request = json.loads(data)
                company_name = request.get("company", "").strip().lower()  # Normalize company name to lowercase
                question = request.get("question", "").strip()

                if not company_name:
                    error_message = "Company name is required."
                    print(f"[Agent 5] Error: {error_message}")
                    await websocket.send_text(json.dumps({"error": error_message}))
                    continue

                # Fetch the file path based on the company name
                I_events_file = COMPANY_FILE_MAP.get(company_name)
                if not I_events_file:
                    error_message = f"Company name '{company_name}' is not recognized."
                    print(f"[Agent 5] Error: {error_message}")
                    await websocket.send_text(json.dumps({"error": error_message}))
                    continue

                # Static file for Agent 3 analysis
                agent_3_file = "trend_data_files/agent_3_output.json"

                # Perform analysis
                analysis_result = await agent4.generate_analysis(I_events_file, agent_3_file, company_name, question)
                print(f"[Agent 5] Analysis Result: {analysis_result}")  # Log analysis result

                # Send the response to the WebSocket client
                await websocket.send_text(json.dumps(analysis_result))

            except json.JSONDecodeError as e:
                error_message = f"JSON decode error: {e}"
                print(f"[Agent 5] Error: {error_message}")
                await websocket.send_text(json.dumps({"error": error_message}))
            except Exception as e:
                error_message = f"Unexpected error: {e}"
                print(f"[Agent 5] Error: {error_message}")
                await websocket.send_text(json.dumps({"error": error_message}))

    except WebSocketDisconnect:
        print("[Agent 5] WebSocket disconnected.")


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
