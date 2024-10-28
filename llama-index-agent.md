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
    Collecting pymupdf
      Downloading PyMuPDF-1.24.12-cp39-abi3-win_amd64.whl.metadata (3.4 kB)
    Downloading PyMuPDF-1.24.12-cp39-abi3-win_amd64.whl (16.0 MB)
       ---------------------------------------- 0.0/16.0 MB ? eta -:--:--
       ---------------------------------------- 0.0/16.0 MB ? eta -:--:--
        --------------------------------------- 0.3/16.0 MB ? eta -:--:--
       - -------------------------------------- 0.8/16.0 MB 1.4 MB/s eta 0:00:11
       -- ------------------------------------- 1.0/16.0 MB 1.5 MB/s eta 0:00:11
       --- ------------------------------------ 1.6/16.0 MB 1.7 MB/s eta 0:00:09
       ----- ---------------------------------- 2.1/16.0 MB 1.8 MB/s eta 0:00:08
       ------ --------------------------------- 2.6/16.0 MB 2.0 MB/s eta 0:00:07
       ------- -------------------------------- 3.1/16.0 MB 2.1 MB/s eta 0:00:07
       --------- ------------------------------ 3.9/16.0 MB 2.3 MB/s eta 0:00:06
       ----------- ---------------------------- 4.5/16.0 MB 2.4 MB/s eta 0:00:05
       ------------- -------------------------- 5.2/16.0 MB 2.5 MB/s eta 0:00:05
       -------------- ------------------------- 5.8/16.0 MB 2.5 MB/s eta 0:00:05
       ----------------- ---------------------- 6.8/16.0 MB 2.7 MB/s eta 0:00:04
       ------------------- -------------------- 7.6/16.0 MB 2.8 MB/s eta 0:00:03
       --------------------- ------------------ 8.7/16.0 MB 2.9 MB/s eta 0:00:03
       ------------------------ --------------- 9.7/16.0 MB 3.1 MB/s eta 0:00:03
       -------------------------- ------------- 10.5/16.0 MB 3.2 MB/s eta 0:00:02
       ---------------------------- ----------- 11.5/16.0 MB 3.3 MB/s eta 0:00:02
       ------------------------------ --------- 12.3/16.0 MB 3.3 MB/s eta 0:00:02
       -------------------------------- ------- 13.1/16.0 MB 3.3 MB/s eta 0:00:01
       ----------------------------------- ---- 14.2/16.0 MB 3.4 MB/s eta 0:00:01
       -------------------------------------- - 15.2/16.0 MB 3.5 MB/s eta 0:00:01
       ---------------------------------------- 16.0/16.0 MB 3.5 MB/s eta 0:00:00
    Installing collected packages: pymupdf
    Successfully installed pymupdf-1.24.12
    

      WARNING: The script pymupdf.exe is installed in 'C:\Users\prabu\AppData\Roaming\Python\Python312\Scripts' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
    


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

# Display excerpts if available
if hasattr(response, 'source_documents') and response.source_documents:
    for i, doc in enumerate(response.source_documents):
        page_info = f"Page {doc.metadata.get('page', 'N/A')}" if doc.metadata else "Unknown page"
        excerpt = doc.text[:500]  # Limit excerpt length to 500 characters
        display(HTML(f'<p style="font-size:16px; color: darkgreen;"><strong>Excerpt from {page_info}:</strong><br>{excerpt}...</p>'))
else:
    display(HTML("<p style='font-size:16px; color: red;'>No excerpts found in the response.</p>"))
```


<p style="font-size:20px; color: darkblue;"><strong>Response:</strong> Tire pressure should be checked monthly when the tires are cold, which means the vehicle has been parked for at least three hours or driven less than 1 mile (1.6 km). This practice is important to ensure proper inflation and safety, especially during cold weather when tire pressure can decrease.</p>



<p style='font-size:16px; color: red;'>No excerpts found in the response.</p>

