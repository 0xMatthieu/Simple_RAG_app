
# Simple_RAG_app

To start add you `openai_api_key`in the secrets.toml file located in .\streamlit folder
Then, just run `streamlit run streamlit_app.py`

## Purpose
The aim is to provide a simple app implementing a RAG function on specific document. It provides a simple Chroma database abstraction layer combined with an simple chat agent to perform RAG

## Structure
```
├── src
│   ├── .streamlit
│   ├── Database
│   │   ├── Private
│   │   │   ├── Chroma_DB
│   │   │   ├── Files
│   │   ├── Public
│   │   │   ├── Chroma_DB
│   │   │   ├── Files
```
## Adding a new document
New document can be added using the following methods
- Adding the document to \Files folder
- Calling the function `add_missing_document_to_chroma_database` or using the button *Add missing files to DB* in the Chroma DB management expander

## Prompt
The prompt can be changed in the app LLM paremeters expander.

## Working with multiple database
Defaut option is using only data from public database
Data from private database can be also used, in this case the app loop through both DB

## working with retrieval agent or tool agent
The POC offers the possibility to use both a retrieval agent or a tool agent. 
It is possible to switch using the Use agent checkbox in App LLM parameteres expander
Differences are the following:
- retrieval agent can only perform RAG on given document. It returns good quality answer.
- tool agent is able to determine and use different tools and provide a custom answer. It can currently use the retrieval agent / a tool to do math / its own knowledge database. Answer quality seems to be a little bit lower on retrieval (way of improvment)