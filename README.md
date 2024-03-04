# Simple_RAG_app
To start add you `openai_api_key`in the secrets.toml file located in .\streamlit folder

Then, just run `streamlit run streamlit_app.py` 

## Purpose
The aim is to provide a simple app implementing a RAG function on specific document. It provides a simple Chroma database abstraction layer combined with an simple chat agent to perform RAG

## Adding a new document
New document can be added using the following methods

 - Adding the document to \Files folder
 - Calling the function `add_missing_document_to_chroma_database` or using the button *Add missing files to DB* in the Chroma DB management expander

## Prompt
The prompt can be changed in the app LLM paremeters expander. 
 
## Working with multiple database
Defaut option is using only data from public database
Data from private database can be also used, in this case the app loop through both DB