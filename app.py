## df-pandas 
## agent - [llm,tools,prompt]
## ui

from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st






_df: pd.DataFrame = None

def set_df(df: pd.DataFrame):
    global _df
    _df = df

if "messages" not in st.session_state:
    st.session_state.messages = []

@tool
def get_data_info(request: str):
    """Get columns, data types, shape, and the first 5 rows of the uploaded CSV."""
    if _df is not None:
        return f""""
        shape: {_df.shape},
        columns and data types: {_df.dtypes.to_string()},
        first 5 rows: {_df.head().to_string()}
    """
    else:
        return "No data available. Please upload a CSV file first."

@tool
def filter_toll(condition: str):
    """Filter rows using a pandas query condition.
    example: "Age > 30 and Department == 'Engineering'' """ 
    if _df is not None:
        try:
            result = _df.query(condition)
            return f"Found {len(result)} rows: {result.to_string()}"
        except Exception as e:
            return f"Error filtering data: {e}"
@tool
def analyze_data(condition:str):
    """Run a pandas expression on the dataframe (referred as to 'df').
    example: 
      df['Age'].mean()
      df.groupby('Department')['Salary'].sum()
      """
    if _df is not None:
        try:
            df = _df
            result = eval(condition)
            return str(result)
        except Exception as e:
            return f"Error analyzing data: {e}"

all_tools = [get_data_info, filter_toll, analyze_data]

llm = ChatGroq(model="openai/gpt-oss-20b")

system_prompt = """You are a data analyst. You have access to a pandas dataframe referred to as 'df'.
You can use the following tools to interact with the dataframe:
-get_data_info: Get columns, data types, shape, and first 5 rows of the uploaded CSV. Pass request="summary".
-filter_toll: Filter rows using a pandas query condition. Example: "Age > 30 and Department == 'Engineering'"
-analyze_data: Run a pandas expression on the dataframe (referred to as 'df').
Always use get_data_info first if you are not sure about the structure of the dataframe.
"""
MEMORY = InMemorySaver()

agent = create_agent(model=llm, tools=all_tools, system_prompt=system_prompt, checkpointer=MEMORY)

def run_agent(query: str):
    res = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "csv-agent"}},
    )
    answer = res['messages'][-1].content
    return answer


## Building a simple Streamlit UI for the CSV Agent
st.header("CSV ANALYST AGENT")
file = st.file_uploader(label = "Upload a CSV file", type = ["csv"])
if file:
    df = pd.read_csv(file)
    set_df(df)
    st.dataframe(df.head())
    st.success(f"CSV file uploaded successfully!{df.shape[0]} rows.")

for mgs in st.session_state.messages:
    role = mgs["role"]
    content = mgs["content"]
    st.chat_message(role).markdown(content)

query  = st.chat_input("Ask a question about the data")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    res = run_agent(query)
    st.chat_message("assistant").markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})


    

