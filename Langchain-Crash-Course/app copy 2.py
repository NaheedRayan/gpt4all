# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here') 


# local_path = ("/home/naheed/Desktop/gpt4all/model/mistral-7b-openorca.Q4_0.gguf") #gpt4all
local_path = ("/home/naheed/Desktop/gpt4all/model/ggml-model-q4_0.gguf") #TinyLlama/TinyLlama-1.1B-Chat-v0.6
# local_path = ("G:\gpt4\model\ggml-model-q4_0.gguf") #TinyLlama/TinyLlama-1.1B-Chat-v0.6



# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
# llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True , streaming=True)
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True ,streaming=True)




from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.agents import create_react_agent, AgentExecutor

from langchain.chains import LLMChain



      

import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

st.title("🔎 LangChain - Chat with search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    

    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True ,streaming=True)

    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

