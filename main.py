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
from Agent_9_card_analysis import OfferWinerAgent
from Agent_9_5_offerwiner_judge import OfferWinerAgentJudge
import logging
from fastapi.staticfiles import StaticFiles
import json

# Warning below is the information of a rapid API which will cost me $29 per month, remember to unsubscribe it after the project is done
# App default-application_10146126
# X-RapidAPI-Key  6f2ffd349dmsh68226b2a11a3c28p1a1851jsn9c0c3be9a4d2
# www.creditcards.com
# GET /creditcard-pointtransfer-transferprogramlist/

from dotenv import load_dotenv
load_dotenv()

FINANCIAL_VARIABLES_JSON = "C:/Users/ymx19/DISCOVER/financial_variables.json"

# Initialize agents with necessary configurations
OPENAI_API_KEY = ""
ALPHAVANTAGE_API_KEY  = "MUW1G1BPCMPUOLWJ"
BASE_URL = "https://www.alphavantage.co/query"

agent1 = Agent1(
    json_file_path=FINANCIAL_VARIABLES_JSON,
    openai_api_key=OPENAI_API_KEY,
)

app = FastAPI()
app.mount("/trend_data_files", StaticFiles(directory="trend_data_files"), name="trend_data_files")
app.mount("/credit_card_data", StaticFiles(directory="credit_card_data"), name="credit_card_data")


# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
market_trends_agent = MarketTrendsAgent()

# Initialize Agent 3,4,5,9
agent3 = Agent3PerformanceAnalysis()
agent4 = Agent4_external_events_AI()
agent5 = Agent5_internal_events_AI()
agent9 = OfferWinerAgent(OPENAI_API_KEY)

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

# =============================================================================
# NEW OFFER WINNER ENDPOINT (POST) - Using input from the frontend
# =============================================================================
# ------------------------------
# WebSocket Endpoint for Agent 9
# ------------------------------
@app.websocket("/ws/agent9")
async def websocket_agent9(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
            except Exception as e:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format", "details": str(e)}))
                continue

            # Validate that the required fields are provided.
            if "card_keys" not in req or "transactions" not in req:
                await websocket.send_text(json.dumps({"error": "Missing 'card_keys' or 'transactions' in the request."}))
                continue

            user_card_keys = req["card_keys"]
            transactions = req["transactions"]

            # Process transactions using Agent 9.
            transaction_results = await agent9.process_transactions(transactions, user_card_keys)

            # Aggregate scores.
            scores = {}
            details = {}
            total_spending = 0
            for res in transaction_results:
                card = res.get("recommended_card", "unknown")
                amount = res.get("amount", 0)
                total_spending += amount
                scores[card] = scores.get(card, 0) + amount
                details.setdefault(card, []).append({
                    "description": res.get("transaction", {}).get("description", ""),
                    "amount": amount
                })

            # Save scores and generate a plot file.
            agent9.save_scores(scores)
            plot_file = agent9.generate_plot_file(scores)

            response = {
                "transaction_results": transaction_results,
                "scores": scores,
                "details": details,
                "plot_file": plot_file
            }
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        logging.info("Agent 9 WebSocket disconnected.")


logging.basicConfig(level=logging.INFO)

# =============================================================================
# Index Endpoint (No cardOptions embedded; client will fetch the JSON)
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def get_index():
    # Return your basic HTML page.
    # The client-side JavaScript (in your index.html) will be responsible for fetching
    # the credit card list from the endpoint (e.g., "/credit_card_data/all_card.json").
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Financial Analysis Chat</title>
      <!-- Include Chart.js and plugin -->
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation/dist/chartjs-plugin-annotation.min.js"></script>
      <style>
         body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f9; margin: 0; padding: 0; }
         .container { max-width: 900px; margin: 20px auto; padding: 20px; background: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
         /* Additional CSS as needed */
      </style>
    </head>
    <body>
      <!-- Your existing HTML for Agents 1-5 goes here -->
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
                 if (data.text_output) content += "<p>" + data.text_output + "</p>";
                 if (data.graph) {
                     content += "<img src='" + data.graph + "' alt='Graph' style='width:100%; max-width:800px;' />";
                 } else {
                     content += "<p style='color: red;'>No graph data received.</p>";
                 }
                 content += "</div>";
                 messages.innerHTML += content;
                 messages.scrollTop = messages.scrollHeight;
             } catch (e) {
                 messages.innerHTML += "<p style='color: red;'>Error processing WebSocket data: " + e.message + "</p>";
             }
         };
         function sendMessage() {
             var company = document.getElementById("company").value.trim();
             var question = document.getElementById("question").value.trim();
             if (!company || !question) {
                 alert("Please enter both a company name and a question.");
                 return;
             }
             ws.send(JSON.stringify({ company: company, question: question }));
         }
      </script>
      
      <!-- Offer Winner Analysis (Agent 9) Section -->
      <div class="container">
          <h2>Offer Winner Analysis</h2>
          <form id="agent9Form" onsubmit="return false;">
              <div class="form-group">
                  <label>Select up to 4 Credit Cards:</label>
                  <div id="cardSelectionContainer">
                      <select id="cardSelect1"></select>
                      <select id="cardSelect2"></select>
                      <select id="cardSelect3"></select>
                      <select id="cardSelect4"></select>
                  </div>
              </div>
              <div class="form-group">
                  <label>Transactions:</label>
                  <div id="transactionsContainer">
                      <div class="transactionRow">
                          <input type="text" class="transCategory" placeholder="Category (e.g., groceries)">
                          <input type="number" class="transAmount" placeholder="Amount">
                          <input type="text" class="transDescription" placeholder="Description (e.g., $150 at Trader Joe's)">
                          <input type="text" class="transMerchant" placeholder="Merchant (e.g., Trader Joe's)">
                      </div>
                  </div>
                  <button type="button" onclick="addTransactionRow()">Add Transaction</button>
              </div>
              <button type="button" onclick="submitAgent9()">Submit Offer Winner Analysis</button>
          </form>
          <div id="agent9Output" class="output" style="display: none;"></div>
          <div id="agent9Graph" class="output" style="display: none;"></div>
      </div>
      
      <!-- Agent 9 Client-Side Code for Fetching Card Options -->
      <script>
         // Use client-side fetching to load the credit card list.
         function fetchCardOptions() {
             fetch("/credit_card_data/all_card.json")
             .then(response => response.json())
             .then(data => {
                 if (!Array.isArray(data)) {
                     console.error("Expected an array but got:", data);
                     return;
                 }
                 data.sort((a, b) => a.localeCompare(b));
                 var optionsHTML = "<option value=''>--Select a Card--</option>";
                 data.forEach(function(cardKey) {
                     optionsHTML += "<option value='" + cardKey + "'>" + cardKey + "</option>";
                 });
                 document.getElementById("cardSelect1").innerHTML = optionsHTML;
                 document.getElementById("cardSelect2").innerHTML = optionsHTML;
                 document.getElementById("cardSelect3").innerHTML = optionsHTML;
                 document.getElementById("cardSelect4").innerHTML = optionsHTML;
             })
             .catch(error => {
                 console.error("Error fetching card options:", error);
             });
         }
         window.addEventListener("load", fetchCardOptions);
         
         // (Keep your Agent 9 WebSocket code and submission functions as-is.)
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
