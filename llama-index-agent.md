```python
!pip install openai
!pip install pymupdf
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: openai in c:\programdata\anaconda3\lib\site-packages (1.52.2)
    Requirement already satisfied: anyio<5,>=3.5.0 in c:\programdata\anaconda3\lib\site-packages (from openai) (4.2.0)
    Requirement already satisfied: distro<2,>=1.7.0 in c:\programdata\anaconda3\lib\site-packages (from openai) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in c:\programdata\anaconda3\lib\site-packages (from openai) (0.27.0)
    Requirement already satisfied: jiter<1,>=0.4.0 in c:\programdata\anaconda3\lib\site-packages (from openai) (0.6.1)
    Requirement already satisfied: pydantic<3,>=1.9.0 in c:\programdata\anaconda3\lib\site-packages (from openai) (2.8.2)
    Requirement already satisfied: sniffio in c:\programdata\anaconda3\lib\site-packages (from openai) (1.3.0)
    Requirement already satisfied: tqdm>4 in c:\programdata\anaconda3\lib\site-packages (from openai) (4.66.5)
    Requirement already satisfied: typing-extensions<5,>=4.11 in c:\programdata\anaconda3\lib\site-packages (from openai) (4.11.0)
    Requirement already satisfied: idna>=2.8 in c:\programdata\anaconda3\lib\site-packages (from anyio<5,>=3.5.0->openai) (3.7)
    Requirement already satisfied: certifi in c:\programdata\anaconda3\lib\site-packages (from httpx<1,>=0.23.0->openai) (2024.8.30)
    Requirement already satisfied: httpcore==1.* in c:\programdata\anaconda3\lib\site-packages (from httpx<1,>=0.23.0->openai) (1.0.2)
    Requirement already satisfied: h11<0.15,>=0.13 in c:\programdata\anaconda3\lib\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: annotated-types>=0.4.0 in c:\programdata\anaconda3\lib\site-packages (from pydantic<3,>=1.9.0->openai) (0.6.0)
    Requirement already satisfied: pydantic-core==2.20.1 in c:\programdata\anaconda3\lib\site-packages (from pydantic<3,>=1.9.0->openai) (2.20.1)
    Requirement already satisfied: colorama in c:\programdata\anaconda3\lib\site-packages (from tqdm>4->openai) (0.4.6)
    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: pymupdf in c:\users\prabu\appdata\roaming\python\python312\site-packages (1.24.12)
    


```python
import nest_asyncio
import os

# Apply nest_asyncio to handle nested event loops (useful for Jupyter notebooks)
nest_asyncio.apply()

# Ensure the OpenAI API key is set as an environment variable
assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

# Import the OpenAI and embedding classes from Llama-Index
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Initialize the language model (LLM) using gpt-4o-mini and embedding model
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
embed_model = OpenAIEmbedding()

# Set the LLM and embedding model globally for usage
Settings.llm = llm
Settings.embed_model = embed_model
```


```python
import fitz  # PyMuPDF
from llama_index.core import Document

# Load the PDF and split by pages
pdf_path = "acura_mdx_manual.pdf"
pdf_document = fitz.open(pdf_path)

# Create a list of Document objects with page-level metadata
acura_docs = []
for page_num in range(len(pdf_document)):
    page = pdf_document[page_num]
    page_text = page.get_text("text")
    document = Document(text=page_text, metadata={"page": page_num + 1})
    acura_docs.append(document)
```


```python
# from llama_index.core import SimpleDirectoryReader

# # Load the Acura MDX manual
# acura_docs = SimpleDirectoryReader(input_files=["acura_mdx_manual.pdf"]).load_data()
```


```python
from llama_index.core import VectorStoreIndex

# Create vector store index from the Acura MDX manual
acura_index = VectorStoreIndex.from_documents(acura_docs)

# Create a query engine for the Acura manual
acura_query_engine = acura_index.as_query_engine(similarity_top_k=3)
```


```python
# Query the Acura MDX manual for tire pressure check recommendations
query = "How often should tire pressure be checked, especially during cold weather?"
response = acura_query_engine.query(query)

# # Print the response attributes to check for 'source_documents'
# print("Response structure:")
# print(response.__dict__)  # Check all attributes of the response

# Display the response and relevant excerpts
from IPython.display import display, HTML

# Display main response
display(HTML(f'<p style="font-size:20px; color: darkblue;"><strong>Response:</strong> {response.response}</p>'))

# Display excerpts from source_nodes
if hasattr(response, 'source_nodes') and response.source_nodes:
    for i, node in enumerate(response.source_nodes):
        page_info = f"Page {node.node.metadata.get('page', 'N/A')}" if node.node.metadata else "Unknown page"
        excerpt = node.node.text[:500]  # Limit excerpt length to 500 characters
        display(HTML(f'<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from {page_info}:</strong><br>{excerpt}...</p>'))
else:
    display(HTML("<p style='font-size:16px; color: red;'>No excerpts found in the response.</p>"))
```


<p style="font-size:20px; color: darkblue;"><strong>Response:</strong> Tire pressure should be checked monthly when the tires are cold. This means the vehicle should have been parked for at least three hours or driven less than 1 mile (1.6 km) before checking the pressure.</p>



<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from Page 487:</strong><br>485
uuWhen DrivinguTire Pressure Monitoring System (TPMS) - Required Federal Explanation
Continued
Driving
Tire Pressure Monitoring System (TPMS) - Required 
Federal Explanation
Each tire, including the spare (if provided), should be checked 
monthly when cold and inflated to the inflation pressure 
recommended by the vehicle manufacturer on the vehicle placard 
or tire inflation pressure label.
(If your vehicle has tires of a different size than the size indicated 
on the vehicle placard or tir...</p>



<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from Page 638:</strong><br>636
Maintenance
Checking and Maintaining Tires
Checking Tires
To safely operate your vehicle, your tires must be of the proper type and size, in 
good condition with adequate tread, and properly inflated.
■Inflation guidelines
Properly inflated tires provide the best combination of handling, tread life, and comfort. 
Refer to the driver’s doorjamb label or specifications page for the specified pressure.
Underinflated tires wear unevenly, adversely affect handling and fuel economy, and 
are more ...</p>



<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from Page 692:</strong><br>690
uuIf a Tire Goes FlatuTemporarily Repairing a Flat Tire
Handling the Unexpected
4. Recheck the air pressure using the gauge 
on the air compressor.
u Do not turn the air compressor on to 
check the pressure.
5. If the air pressure is:
• Less than 19 psi (130 kPa):
Do not add air or continue driving. The 
leak is too severe. Call for help and have 
your vehicle towed.
2 Emergency Towing P. 724
• 33 psi (230 kPa) or more:
Continue driving for another five 
minutes or until you reach the neares...</p>



```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker

# Define the query engine tool for Acura manual
query_engine_tools = [
    QueryEngineTool(
        query_engine=acura_query_engine,
        metadata=ToolMetadata(
            name="acura_manual",
            description="Provides information from the Acura MDX 2022 owner's manual",
        ),
    )
]

# Create a function-calling agent worker
agent_worker = FunctionCallingAgentWorker.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)

# Convert the agent worker to an agent
agent = agent_worker.as_agent()

# Use the agent to ask a question about the Acura manual
response = agent.chat("How often should tire pressure be checked, especially during cold weather?")
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```

    Added user message to memory: How often should tire pressure be checked, especially during cold weather?
    === Calling Function ===
    Calling function: acura_manual with args: {"input": "tire pressure check frequency cold weather"}
    === Function Output ===
    Tire pressure should be checked monthly when the tires are cold. This means the vehicle should have been parked for at least three hours or driven less than 1 mile (1.6 km) before checking the pressure. Regular checks are especially important in cold weather, as temperatures can cause tire pressure to drop.
    === LLM Response ===
    Tire pressure should be checked monthly, especially during cold weather. It's best to check the pressure when the tires are cold, meaning the vehicle should have been parked for at least three hours or driven less than 1 mile (1.6 km) before checking. Cold temperatures can cause tire pressure to drop, making regular checks crucial.
    


<p style="font-size:20px">Tire pressure should be checked monthly, especially during cold weather. It's best to check the pressure when the tires are cold, meaning the vehicle should have been parked for at least three hours or driven less than 1 mile (1.6 km) before checking. Cold temperatures can cause tire pressure to drop, making regular checks crucial.</p>


# Agentic Architecture Overview

The setup using `FunctionCallingAgentWorker` with Llama Index can be considered an example of agentic architecture. Here's a breakdown of why this approach qualifies:

1. **Agents and Autonomy**  
   - The `FunctionCallingAgentWorker` creates an agent that autonomously decides which tools (query engines) to utilize based on the user's query.
   - This makes it an "agent" because it can perform actions independently to resolve queries. For instance, the agent autonomously decides which section of the manual to query to answer a question.

2. **Tool Integration**  
   - The agent is integrated with "tools" (`QueryEngineTool`), which provide specific capabilities—in this case, querying the Acura manual data.
   - This tool integration is central to agentic architecture as it allows the agent to perform specialized tasks using pre-defined functionalities.

3. **Reasoning and Function Calling**  
   - The `FunctionCallingAgentWorker` allows the agent to reason and call specific functions as needed based on the user's prompt.
   - This setup enables the agent to make decisions and take actions, such as querying the Acura manual for specific information like resetting the oil change light.

## Differences from a Basic Query System
- A basic query system only returns search results without processing or reasoning, while the agentic approach "thinks through" the required steps.
- Agentic architecture enables multiple decision-making steps and tool usage, adding sophistication beyond simple query-response mechanisms.

## Benefits
- **Modularity**: Additional tools can be added to the agent, enabling it to autonomously decide when to use each one.
- **Scalability**: The agent can scale to handle complex, multi-step queries and interactions, making it more versatile than a basic query engine.

In summary, this setup leverages principles of agentic architecture, enabling it to dynamically and autonomously interact with users' queries. This is beneficial for scenarios that require more than simple responses, making it capable of sophisticated, contextualized interactions.


```python
# Query the Acura MDX manual for tire pressure check recommendations
query = "How often should tire pressure be checked, especially during cold weather?"
response = acura_query_engine.query(query)

# Display the response and relevant excerpts
from IPython.display import display, HTML

# Display main response
display(HTML(f'<p style="font-size:20px; color: darkblue;"><strong>Response:</strong> {response.response}</p>'))

# Display excerpts from source_nodes if available
if hasattr(response, 'source_nodes') and response.source_nodes:
    for i, node in enumerate(response.source_nodes):
        page_info = f"Page {node.node.metadata.get('page', 'N/A')}" if node.node.metadata else "Unknown page"
        excerpt = node.node.text[:500]  # Limit excerpt length to 500 characters for readability
        display(HTML(f'<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from {page_info}:</strong><br>{excerpt}...</p>'))
else:
    display(HTML("<p style='font-size:16px; color: red;'>No excerpts found in the response.</p>"))
```


<p style="font-size:20px; color: darkblue;"><strong>Response:</strong> Tire pressure should be checked monthly when the tires are cold. It's also advisable to check the pressure before long trips. Cold tires mean the vehicle has been parked for at least three hours or driven less than 1 mile.</p>



<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from Page 487:</strong><br>485
uuWhen DrivinguTire Pressure Monitoring System (TPMS) - Required Federal Explanation
Continued
Driving
Tire Pressure Monitoring System (TPMS) - Required 
Federal Explanation
Each tire, including the spare (if provided), should be checked 
monthly when cold and inflated to the inflation pressure 
recommended by the vehicle manufacturer on the vehicle placard 
or tire inflation pressure label.
(If your vehicle has tires of a different size than the size indicated 
on the vehicle placard or tir...</p>



<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from Page 638:</strong><br>636
Maintenance
Checking and Maintaining Tires
Checking Tires
To safely operate your vehicle, your tires must be of the proper type and size, in 
good condition with adequate tread, and properly inflated.
■Inflation guidelines
Properly inflated tires provide the best combination of handling, tread life, and comfort. 
Refer to the driver’s doorjamb label or specifications page for the specified pressure.
Underinflated tires wear unevenly, adversely affect handling and fuel economy, and 
are more ...</p>



<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from Page 692:</strong><br>690
uuIf a Tire Goes FlatuTemporarily Repairing a Flat Tire
Handling the Unexpected
4. Recheck the air pressure using the gauge 
on the air compressor.
u Do not turn the air compressor on to 
check the pressure.
5. If the air pressure is:
• Less than 19 psi (130 kPa):
Do not add air or continue driving. The 
leak is too severe. Call for help and have 
your vehicle towed.
2 Emergency Towing P. 724
• 33 psi (230 kPa) or more:
Continue driving for another five 
minutes or until you reach the neares...</p>



```python
from datetime import datetime, timedelta

# Set the last tire pressure check date to 40 days prior to today's date
last_check_date = datetime.now() - timedelta(days=40)
check_interval_days = 30  # Recommended interval for tire pressure checks in days

# Display the last check date and the check interval for verification
print(f"Today's date: {datetime.now().strftime('%Y-%m-%d')}")
print(f"Last tire pressure check date: {last_check_date.strftime('%Y-%m-%d')}")
print(f"Recommended check interval: {check_interval_days} days")
```

    Today's date: 2024-10-28
    Last tire pressure check date: 2024-09-18
    Recommended check interval: 30 days
    


```python
import os
import requests

def get_current_temperature(city):
    """Fetches the current temperature for a given city using WeatherAPI."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise ValueError("WEATHER_API_KEY environment variable is not set.")
    
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data['current']['temp_f']
        return temperature
    else:
        print(f"Error fetching weather data: {response.status_code}")
        return None

# Test the function with "Powell" as the city
city = "Powell"
current_temp = get_current_temperature(city)
if current_temp is not None:
    print(f"The current temperature in {city} is {current_temp}°F.")
else:
    print("Failed to fetch the current temperature.")
```

    The current temperature in Powell is 48.4°F.
    


```python
def should_send_reminder(last_check_date, check_interval_days, current_temp, temperature_threshold=32):
    """Determine if a tire pressure reminder should be sent based on check interval and temperature."""
    days_since_last_check = (datetime.now() - last_check_date).days
    print(f"Days since last tire pressure check: {days_since_last_check}")
    print(f"Current temperature in Powell: {current_temp}°F")
    
    # Check if either the check interval has passed or the temperature is below the threshold
    if days_since_last_check >= check_interval_days or current_temp < temperature_threshold:
        return True
    return False

# Test the function with current values
temperature_threshold = 50  # Set temperature threshold in Fahrenheit
reminder_needed = should_send_reminder(last_check_date, check_interval_days, current_temp, temperature_threshold)

if reminder_needed:
    print("Reminder conditions met: Time to check tire pressure.")
else:
    print("No reminder needed at this time.")
```

    Days since last tire pressure check: 40
    Current temperature in Powell: 48.4°F
    Reminder conditions met: Time to check tire pressure.
    


```python
# from twilio.rest import Client
# import os

# def send_sms_reminder(message, to_phone):
#     """Sends an SMS reminder via Twilio using API Key SID and Secret."""
#     api_key_sid = os.getenv("TWILIO_API_KEY_SID")
#     api_key_secret = os.getenv("TWILIO_API_KEY_SECRET")
#     account_sid = os.getenv("TWILIO_ACCOUNT_SID")  # Still need the Account SID
#     from_phone = os.getenv("TWILIO_PHONE_NUMBER")
    
#     if not all([api_key_sid, api_key_secret, account_sid, from_phone]):
#         raise ValueError("Twilio environment variables are not set properly.")

#     client = Client(api_key_sid, api_key_secret, account_sid)
    
#     message = client.messages.create(
#         body=message,
#         from_=from_phone,
#         to=to_phone
#     )
#     return message.sid

# # Example message content
# reminder_message = (
#     f"Reminder: It’s time to check your tire pressure. The temperature in Powell is {current_temp}°F, "
#     "which can impact tire pressure."
# )

# # Send SMS if reminder conditions are met
# if reminder_needed:
#     sms_sid = send_sms_reminder(reminder_message, os.getenv("USER_PHONE_NUMBER"))
#     print(f"SMS sent with SID: {sms_sid}")
# else:
#     print("No SMS sent; conditions not met.")
```
