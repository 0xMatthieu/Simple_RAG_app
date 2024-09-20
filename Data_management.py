
# this software is manly derivated from the example found at this link https://github.com/rubentak/Langchain/blob/main/notebooks/Langchain_doc_chroma.ipynb
print("Start to load data mgt")
import time
start_time = time.time()

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory

from langchain.chains import LLMMathChain

from langchain import hub

from langchain.retrievers import MergerRetriever
from langchain.tools.retriever import create_retriever_tool


from streamlit import secrets
from Chroma_db_management import ChromaDb

print(f"Time to load data mgt main lib: {time.time() - start_time}")


"""
try:
    from traceloop.sdk import Traceloop
    Traceloop.init(app_name="chroma_app",api_key=secrets['llmetry_api_key'], disable_batch=True)
except ImportError:
    print("Traceloop not available")
"""

#streamlit bug fix
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    print("pysqlite not available")


class ChatController(object):
    def __init__(self):
        print("Start to load chat agent")
        start_time = time.time()
        self._create_chat_agent()
        self.init_qa_function()
        print(f"Time to load chat agent: {time.time() - start_time}")
        
    def _create_chat_agent(self):

        # init some variables
        self.api_key = secrets['openai_api_key'] # open ai api key shall be created in openai website
        self.use_private_data = False
        self.llm_model = "gpt-4o-mini"#"gpt-4o-2024-08-06"
        self.retriever_output_number = 20 #default value is 4
        self.vectordb = ChromaDb(txt_directory = './Database/Public/Files/', chroma_db_directory = './Database/Public/Chroma_DB/')
        self.vectordb_private = ChromaDb(txt_directory = './Database/Private/Files/', chroma_db_directory = './Database/Private/Chroma_DB/')
        self.vectordb_raptor = ChromaDb(txt_directory = './Database/Raptor/Files/', chroma_db_directory = './Database/Raptor/Chroma_DB/')
        self.rag_prompt = None
        self.use_agent = True
        self.use_Raptor = False
        self.update_llm = ""
    
    def init_qa_function(self):
        
        # retriever function can be test with this kind of example: docs = retriever.get_relevant_documents("How to use TTC 500 external RAM")
        retriever = self.vectordb.vectordb.as_retriever(search_kwargs={"k": self.retriever_output_number})
        retriever_private = self.vectordb_private.vectordb.as_retriever(search_kwargs={"k": self.retriever_output_number})
        retriever_raptor = self.vectordb_raptor.vectordb.as_retriever(search_kwargs={"k": self.retriever_output_number})
        
        # Set up the turbo LLM
        turbo_llm = ChatOpenAI(
            openai_api_key = self.api_key,
            temperature=0,
            model_name=self.llm_model
        )
        
        if self.use_private_data == True:
            # We just pass a list of retrievers.
            retriever = MergerRetriever(retrievers=[retriever, retriever_private])
        if self.use_Raptor == True:
            retriever = retriever_raptor                                  

        math_chain = LLMMathChain.from_llm(llm=turbo_llm)

        Document_tool = create_retriever_tool(retriever, "document_tool",
            "Use this tool when you need to answer question about technical products."
            )

        tools = [
            Document_tool, 
            
            Tool(
                name="math_chain",
                func=math_chain.run,
                description="useful when you need to do some math calculation."
            ),
        ]
        
        #model_with_tools = turbo_llm.bind_tools(tools)
        
        # tuto : https://python.langchain.com/docs/how_to/agent_executor/
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_tool_calling_agent(turbo_llm, tools, prompt)

        self.chat_agent = AgentExecutor(
            agent=agent, tools=tools,
            verbose=True, 
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            handle_parsing_errors=True
        )
             
        if self.use_agent == False:
            # Create the chain to answer questions
            prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            question_answer_chain = create_stuff_documents_chain(turbo_llm, prompt)
            qa_chain = create_retrieval_chain(retriever, question_answer_chain)
        
            self.chat_agent = qa_chain

    # wrapper function
    def update_llm_model(self, text):
        print(f"update {text}")
        self.update_llm = self.update_llm + " / " + text

    # function needed to bypass streamlit limitation
    def do_update(self):
        if self.update_llm_model != "":
            self.init_qa_function()
            self.update_llm = ""
            print(f"------------------------------")
            #print(self.chat_agent)
            print(self.llm_model)
            print(self.use_agent)
            print(self.retriever_output_number)

    
    def ask(self, query = ""):
        
        if self.use_agent == False:
            llm_response = self.chat_agent.invoke({"input": query})
            print(llm_response['answer'])
            llm_response = llm_response['answer']
        else:
            llm_response = self.chat_agent.invoke({"input": query})
            print(llm_response)
            llm_response = llm_response['output']
        
        return llm_response
        
        
if __name__ == "__main__":
    chat = ChatController()
    #chat.use_agent = True
    #chat.use_Raptor = True
    #chat.vectordb_raptor.add_missing_document_to_chroma_database('Raptor')
    #chat.init_qa_function()
    print("done")
    # raptor test
    #chat.ask('how to use analog inputs on the TTC500 in C ? Provide an example')
    #chat.ask("How to use TTC 500 external RAM in C ?")
    #chat.ask('how to use PWM outputs on the TTC500 in C ?')
    #chat.rag_chain.invoke('how to use PWM outputs on the TTC500 in C ?')
    #chat.ask('can you do 5 x 5 + 2 x 2 ?')
    #chat.ask('what combinations can be done with HDA 7000 ?')
    #chat.ask("How to use IO link on HMG 4000 ? Can it be used on HMG 3010 too ?")
    #chat.vectordb.add_or_delete_a_document(filepath = 'PDF_private\\Configurator.xlsx', method = 'Delete')
    #chat.vectordb.add_or_delete_a_document(filepath = 'PDF_private\\2300_PxPanic() and PxAbort().txt', method = 'Delete')
    #chat.vectordb.list_all_chromadb_files()
    #chat.vectordb.add_missing_document_to_chroma_database()
    #chat.vectordb.add_a_directory_to_chroma_database(txt_directory = chat.vectordb.txt_directory)
