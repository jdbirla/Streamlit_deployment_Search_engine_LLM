import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import SystemMessage
from dotenv import load_dotenv
import os

## just re pushing
# Load environment variables
load_dotenv()

# Streamlit UI
st.title("🔎 LangChain - Chat with Search")

"""
In this example, we're using `StreamlitCallbackHandler` to display
the thoughts and actions of an agent in an interactive Streamlit app.
"""

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize session
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Wait for user input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True,
    )

    # Define tools
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    search = DuckDuckGoSearchRun(name="Search")

    tools = [search, arxiv, wiki]

    # 🧠 Create agent using the new API
    system_message = SystemMessage(
        content="You are a helpful AI assistant who answers user questions using available tools."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        system_message,
        MessagesPlaceholder(variable_name="input")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Stream response in Streamlit
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = agent_executor.invoke(
            {"input": [{"role": "user", "content": prompt}]},
            config={"callbacks": [st_cb]}
        )

        answer = response.get("output", "No response.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
