
# this software is manly derivated from the example found at this link https://github.com/rubentak/Langchain/blob/main/notebooks/Langchain_doc_chroma.ipynb

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, initialize_agent, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain

# RAG prompt
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.tools.base import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.retrievers import MergerRetriever

import streamlit as st

from Chroma_db_management import ChromaDb

"""
try:
    from traceloop.sdk import Traceloop
    Traceloop.init(app_name="chroma_app",api_key=st.secrets['llmetry_api_key'], disable_batch=True)
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
        self._create_chat_agent()
        self.init_qa_function()
        
    def _create_chat_agent(self):

        # init some variables
        self.api_key = st.secrets['openai_api_key'] # open ai api key shall be created in openai website
        self.use_private_data = False
        self.llm_model = "gpt-3.5-turbo-0125"#"gpt-4-turbo-preview"
        self.retriever_output_number = 4 #default value is 4
        self.vectordb = ChromaDb(txt_directory = './Database/Public/Files/', chroma_db_directory = './Database/Public/Chroma_DB/')
        self.vectordb_private = ChromaDb(txt_directory = './Database/Private/Files/', chroma_db_directory = './Database/Private/Chroma_DB/')
        self.agent_prompt = """Assistant is a large language model trained by OpenAI.

            Respond to the human as helpfully and accurately as possible. You have access to the following tools:

            {tools}

            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
            Valid "action" values: "Final Answer" or {tool_names}

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {{
              "action": $TOOL_NAME,
              "action_input": $INPUT
            }}
            ```

            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {{
              "action": "Final Answer",
              "action_input": "Final response to human"
            }}

            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly and use your own knowledge if appropriate. 
            Format is Action:```$JSON_BLOB```then Observation 
            """
        self.retrieval_prompt = """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            
            Use the maximum sentences you need to provide accurate and detailed answers to diverse queries.

            Question: {question}
            
            Context: {context}

            Answer:
            """
        self.rag_prompt = None
        self.use_agent = False
        self.message_history = []
        self.rag_chain = None
        self.use_RAG = False
        self.update_llm = ""
    
    def init_qa_function(self):
        
        # retriever function can be test with this kind of example: docs = retriever.get_relevant_documents("How to use TTC 500 external RAM")
        retriever = self.vectordb.vectordb.as_retriever(search_kwargs={"k": self.retriever_output_number})
        retriever_private = self.vectordb_private.vectordb.as_retriever(search_kwargs={"k": self.retriever_output_number})
        
        # Set up the turbo LLM
        turbo_llm = ChatOpenAI(
            openai_api_key = self.api_key,
            temperature=0,
            model_name=self.llm_model
            #max_tokens = 4096 - 500
        )

        PROMPT = PromptTemplate(
            template=self.retrieval_prompt, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"verbose": True, "prompt": PROMPT}


        #RAG
        prompt = hub.pull("rlm/rag-prompt")
        self.rag_prompt = prompt.messages[0].prompt.template

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | turbo_llm
            | StrOutputParser()
        )
        
        
        if self.use_private_data == True:
            # We just pass a list of retrievers.
            retriever = MergerRetriever(retrievers=[retriever, retriever_private])
            
          
        # Create the chain to answer questions
        qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                          chain_type="stuff",
                                          retriever=retriever,
                                          chain_type_kwargs=chain_type_kwargs,
                                          return_source_documents=False,
                                          verbose=True)
                                            

        math_chain = LLMMathChain.from_llm(llm=turbo_llm)

        class Document_inputs(BaseModel):
            """Inputs to the document tool."""

            input: str = Field(
                description="query to look up in tool"
            )
               
        tools = [
            StructuredTool(
                name="Document tool",
                func=qa_chain.invoke,
                description="Use this tool when you need to answer question about technical products.",
                args_schema=Document_inputs
            ), 
            
            Tool(
                name="math chain",
                func=math_chain.run,
                description="useful when you need to do some math calculation."
            ),
        ]
        
        prompt = hub.pull("hwchase17/structured-chat-agent")
        
        prompt.messages[0].prompt.template = self.agent_prompt
        agent = create_structured_chat_agent(
            llm=turbo_llm,
            tools=tools,
            prompt=prompt
        )
        
        
        self.chat_agent = AgentExecutor(
            agent=agent, tools=tools,
            verbose=True, 
            #memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            handle_parsing_errors=True
        )
        
        if self.use_agent == False:
            self.chat_agent = qa_chain
        # not used
        if self.use_RAG == True:
            self.chat_agent = self.rag_chain


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
            llm_response = self.chat_agent.invoke(query)
            print(llm_response)
            llm_response = llm_response['result']
        else:
            llm_response = self.chat_agent.invoke({"input": query})
            print(llm_response)
            llm_response = llm_response['output']
            #length = len(llm_response.get('chat_history')) - 1
            #llm_response = llm_response.get('chat_history')[length].content
        
        return llm_response
        
        
if __name__ == "__main__":
    chat = ChatController()
    chat.use_agent = True
    chat.init_qa_function()
    print("done")
    
    
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
