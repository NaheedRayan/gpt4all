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


local_path = ("/home/naheed/Desktop/gpt4all/model/mistral-7b-openorca.Q4_0.gguf") #gpt4all
# local_path = ("/home/naheed/Desktop/gpt4all/ggml-model-q4_0.gguf") #TinyLlama/TinyLlama-1.1B-Chat-v0.6
# local_path = ("G:\gpt4\model\ggml-model-q4_0.gguf") #TinyLlama/TinyLlama-1.1B-Chat-v0.6

import os
import pprint
os.environ["SERPER_API_KEY"] = "e1f8cd9de27d3b78f9ab3428167883002e76c33b"

from langchain.utilities import GoogleSerperAPIWrapper
search = GoogleSerperAPIWrapper()


# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
# llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True , streaming=True)
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)



from langchain.tools import Tool
from langchain.chains import LLMMathChain

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.agents import create_react_agent, AgentExecutor

from langchain.chains import LLMChain



llm_math_chain = LLMMathChain(llm=llm, verbose=True)

google_search = Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events",
        # description="useful for when you need to answer questions about current news",
        # coroutine= ... <- you can specify an async method if desired as well
    )

calculator =  Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        # coroutine= ... <- you can specify an async method if desired as well
    )

tools = [ google_search , calculator ]


# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

# Set up the base template
template_asking_with_tools = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! 

New question: {question}
{agent_scratchpad}"""


template_for_comapring_answers = """Answer the following questions as best you can.


Use the following format:

Question: the input question you must answer
Final Answer: the final answer to the original input question


Promt1: "{llm_generated}" 
Promt2: "{actual_answer}"

Begin! 

Question: "{question}" Does Promt1 answer matches with Promt2 . Answer it in only Yes or No.
"""



prompt1_with_tools = PromptTemplate(
    template=template_asking_with_tools,
    tools=tools,
    input_variables=["question"]
)

prompt2_for_comparison = PromptTemplate(
    template=template_for_comapring_answers,
    input_variables=["question", "llm_generated" , "actual_answer"]
)

tool_names = [tool.name for tool in tools]
agent = create_react_agent(

        tools=tools ,
        prompt=prompt1_with_tools,
        llm = llm,
    )


agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True,handle_parsing_errors=True,max_iterations=3)
       




# Show stuff to the screen if there's a prompt
if prompt: 
   
    script = llm_generated = agent_executor.invoke({"question": prompt})

    st.write(script) 

    # with st.expander('Title History'): 
    #     st.info(title_memory.buffer)

    # with st.expander('Script History'): 
    #     st.info(script_memory.buffer)

    # with st.expander('Wikipedia Research'): 
    #     st.info(wiki_research)
