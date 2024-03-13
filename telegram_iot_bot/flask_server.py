from flask import Flask, request
import os
import datetime
import requests


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI , ChatOpenAI

from langchain.agents import tool
from langchain.tools import Tool

# from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchRun

# from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.chains import LLMMathChain

from langchain.chains.conversation.memory import  ConversationBufferWindowMemory

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain.agents import AgentExecutor



# api keys
# OPENAI_API_KEY = "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"    #naheed
OPENAI_API_KEY = "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"    #saiful
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# instantiating llm and the model
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0)


# component name and status
component_dict = {
    "light1":0,
    "light2":0,
    "fan1":0,
    "fan2":0,
}






# schema for building format for instructions
# --------------------------------------------------------------------------------------------------------
component_name_schema = ResponseSchema(name="component_name",
                             description="This is the name of the component or device")

reasoning_schema = ResponseSchema(name="reasoning",
                                      description="Just reasoning")

component_status_schema = ResponseSchema(name="component_status",
                                    description="This is an integer score between 0-9")

response_schemas = [component_name_schema, 
                    reasoning_schema,
                    component_status_schema]

# building structured output parser for parsing the json part from the output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions() # formated instructions will be injected to the prompt1


# --------------------------------------------------------------------------------------------------------



# prompt for triggering iot component
prompt1 = PromptTemplate.from_template("""

Select the tools as best as you can. You can also ignore if not possible. 

{chat_history}
Examples : 
                                      
Question: Switch On the light1
component_name: light1
reasoning: we have to switch it on
component_status: 9                                  

Question: Switch off the light1
component_name: light1
reasoning: we have to switch it off
component_status: 0

Question: Increase the fa1 speed to 5
component_name: fa1
reasoning: we have to set the level to 5
component_status: 5       

Question: Slow the fa2 speed to 2
component_name: fa2
reasoning: we have to set the level to 2
component_status: 2  

                                                          
Read the Examples and answer Question1 using the following format for the output:
{format_instructions}                                                         
                    
Begin!
        
Question1 : {question}
                             
""")

# memory for context
memory = ConversationBufferWindowMemory(memory_key="chat_history" , k=1 , return_messages=True)


# -----------------------------------------------Functions---------------------------------------------------------

# function for calling the iot component
def call_iot_component(response_as_dict):


    ip_addr = '192.168.43.120'
    component_name = 'fan1'
    component_status = '0'

    component_name = response_as_dict['component_name']
    component_status =response_as_dict['component_status']

    url = f'http://{ip_addr}/?component_name={component_name}&component_status={component_status}'

    try:
        

        response = requests.get(url)
        if response.status_code == 200:
            print("Request successful")
            print(response.text)
            return response.text
        else:
            print("Request failed with status code:", response.status_code)
            return "Request failed with status code:"+ response.status_code
    except requests.RequestException as e:
        print("Error:", e)
        return e

 

# function for processing the iot device trigger data
def iot(data):
    print("+++++++++ "+data+" ++++++++++")

    chain = prompt1 | llm 
    response = chain.invoke({'question':data , 'format_instructions':format_instructions , 'chat_history':memory })

    # parsing the response as a dict
    response_as_dict = output_parser.parse(response.content)
    print(response_as_dict)

    if response_as_dict['component_name'] in component_dict:
        print('Component Exists')
        response_as_dict['exist'] = True
        response_as_dict['status'] = 'Successful'
        x = call_iot_component(response_as_dict)

        # response_as_dict['status_message'] = 'The component exist'
        response_as_dict['status_message'] = x


    else:
        print('Component Does not Exist')
        response_as_dict['exist'] = False
        response_as_dict['status'] = 'Failed'
        response_as_dict['status_message'] = 'The component does not exist'


    # clearing memory
    memory.clear()
    
    return response_as_dict


def time_date(data):
    current_date_and_time = datetime.datetime.now().replace(microsecond=0)
    print("Current Date and Time:", current_date_and_time)
    return str(current_date_and_time)


# ------------------------------------------------------Tools----------------------------------------------------------

# for general question answering
prompt2 = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt2)
ddg_search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())



llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# initialize the LLM tool
llm_tool = Tool(
    name='Language_Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)


duckduckgo_search = Tool.from_function(
        func=ddg_search.run,
        name="DuckDuckGo_Search",
        description="useful for when you need to answer questions about current news and when the user asks for searching from the web",
        # description="useful for when you need to answer questions about current news",
        # coroutine= ... <- you can specify an async method if desired as well
    )




calculator =  Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        # coroutine= ... <- you can specify an async method if desired as well
    )


time_and_date =  Tool.from_function(
        func=time_date,
        name="TimeAndDate",
        description="useful for when you need to answer Time and Date",
        # coroutine= ... <- you can specify an async method if desired as well
    )



iot_device =  Tool.from_function(
        func=iot,
        name="IOT_device",
        description="""useful for when you need to answer these types of questions about components
        Switch On the [Device Name]
        Switch off the [Device Name]
        Increase the [Device Name] speed to 5
        Slow the [Device Name] speed to 2
        Activate [Device Name]
        Switch on the [Device Name]
        Power up the [Device Name]
        Start [Device Name]
        Initiate [Device Name]
        Engage [Device Name]
        Turn on the [Device Name]
        Activate the power to [Device Name]
        Bring [Device Name] online
        Enable [Device Name]

        and always pass the full input sentence
      
        """,
        # coroutine= ... <- you can specify an async method if desired as well
    )

tools = [llm_tool, duckduckgo_search ,calculator ,iot_device , time_and_date]



# ---------------------------------Prompt Agent AgentExecutor-------------------------------------------------------------------------------------------


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
llm_with_tools = llm.bind_tools(tools)
# memory.load_memory_variables({})['chat_history']



agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history":lambda x :(memory.load_memory_variables({})['chat_history'])
    }

    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)



agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True , memory=memory)

memory.clear()

# ---------------------------------------------------------------Main Flask APP-----------------------------------------------------------------------




app = Flask(__name__)

@app.route('/', methods=['POST'])
def run_script():
    if request.method == 'POST':
        # Retrieve data from the POST request
        
        data = request.data.decode('utf-8')
    
        print("+++++++++++++++++++++++++")
        result = execute_script(data)
        
        print(result)
        return result

def execute_script(data):
    # This function executes your Python script
  
    try:
        response =  agent_executor.invoke({"input": data})
    except:
        return "Something went wrong"
    
    return response["output"]

if __name__ == '__main__':
    app.run(debug=True)
