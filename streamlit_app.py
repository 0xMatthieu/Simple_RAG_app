import streamlit as st
import pandas as pd
from Data_management import ChatController
from importlib.metadata import version


def app_init():
    # Initialize session_state if it doesn't exist
    if 'chat' not in st.session_state:
        st.session_state['chat'] = ChatController()
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

def library_version_display():
    library_version_expander = st.expander(label='Current library versions used')

    with library_version_expander:
        st.write('streamlit version is ' + version('streamlit'))
        st.write('openai version is ' + version('openai'))
        st.write('langchain version is ' + version('langchain'))
        st.write('langchain_community version is ' + version('langchain_community'))
        
def write_files_trained_on():
    files_trained_on_expander = st.expander(label='App was trained on following files:')
    files = st.session_state.chat.vectordb.list_all_chromadb_files()
    files_2 = st.session_state.chat.vectordb_private.list_all_chromadb_files()
    
    with files_trained_on_expander:
        # list all files in directory
        st.write("# public database")
        for filename in files:
            st.write(f"{filename}")
        st.write()
        st.write("# private database")
        for filename in files_2:
            # display the file name in the left column
            st.write(f"{filename}")
        
def app_parameters():
    param_expander = st.expander(label='App llm parameters')
    
    with param_expander:
        # Adding a select button
        st.session_state.chat.llm_model = st.selectbox(
                label = 'Choose a model:',
                options = ('gpt-3.5-turbo-0125', 'gpt-4-turbo-preview'),
                index = 0,
                on_change =st.session_state.chat.update_llm_model, args =("llm model",))
               
        st.write('You selected:', st.session_state.chat.llm_model)
        st.write('GPT4 is currently 20x more expensive than GPT3.5')
            
        st.session_state.chat.use_agent = st.checkbox(label='Use agent', 
            value=st.session_state.chat.use_agent,
            on_change =st.session_state.chat.update_llm_model, args =("use agent",))

        integer_value = st.number_input(
            label='Number of documents returned by the retriever, default value is 4:',
            min_value=0,  # Minimum value
            max_value=100,  # Maximum value
            value=st.session_state.chat.retriever_output_number,  # Default value from session state
            step=1,  # Step size
            format='%d'  # '%d' as integer
        )
        # Check if the value has changed
        if integer_value != st.session_state.chat.retriever_output_number:
            st.session_state.chat.retriever_output_number = integer_value  # Update the session state
            st.session_state.chat.update_llm_model("retriever output number")
            
        # Retrieval prompt
        st.session_state.chat.retrieval_prompt = st.text_area("retrieval prompt", st.session_state.chat.retrieval_prompt)

        # Agent prompt
        st.session_state.chat.agent_prompt = st.text_area("agent prompt", st.session_state.chat.agent_prompt)

        if st.button('Modify'):
            st.session_state.chat.update_llm_model("update prompt")

    st.session_state.chat.do_update() 

def manage_chroma_db():
    db_expander = st.expander(label='Chroma DB management')
    
    with db_expander:

        # test to use more than one DB and a merge retriever function
        st.session_state.chat.use_private_data = st.checkbox(label='Use private database', 
            value=st.session_state.chat.use_private_data,
            on_change =st.session_state.chat.update_llm_model, args =("use private data",))
        
        # Adding a select button to choose a DB
        selected_db = st.selectbox(
                label = 'Choose a DB:',
                options = ('public', 'private'),
                index = 0)
        
        st.write('You selected the following db:', selected_db)

        match selected_db:
                case 'public':
                    vector_db = st.session_state.chat.vectordb
                case 'private':
                    vector_db = st.session_state.chat.vectordb_private

        # button to add missing files
        if st.button('Add missing files to DB'):
            vector_db.add_missing_document_to_chroma_database()
                
        # Delete a document
        path_to_doc_to_delete = st.text_input("Document to delete (case sensitive)")

        # button to delete a document
        if st.button('Delete document'):
            vector_db.add_or_delete_a_document(filepath = path_to_doc_to_delete, method = 'Delete')

    st.session_state.chat.do_update() 


# This is the title of the app
st.title('Simple RAG App')

# layout and param
app_init()

with st.sidebar:
    library_version_display()
    write_files_trained_on()
    manage_chroma_db()
    app_parameters()
    


# User types their message into the text_input
user_input = st.text_input("Type your message here")

if st.button('Send'):
    # Add user's message to chat history
    st.session_state.chat_history.append(f"User: {user_input}")

    # tokenize the new input sentence
    st.session_state.chat.message_history.append({"role": "user", "content": f"{user_input}"})

    # Here, instead of echoing the user's message, you would 
    # call your chatbot function and generate a response
    reply_content = st.session_state.chat.ask(user_input)
    
    # append the new reply to the message history
    st.session_state.chat.message_history.append({"role": "assistant", "content": f"{reply_content}"}) 
    
    bot_response = f"Assistant: {reply_content}"
    st.session_state.chat_history.append(f"{bot_response}")

# Assuming that chat history is arranged in question-answer-question-answer...
for i in range(len(st.session_state.chat_history)-1, -1, -2):
    st.markdown(f"<p style='color:green'>Q: {st.session_state.chat_history[i-1]}</p>", unsafe_allow_html=True)
    st.write(f"A: {st.session_state.chat_history[i]}")
    
# Button to clear chat history
if st.button('Clear chat history'):
    st.session_state.chat_history = []
    st.session_state.chat.message_history = []

st.write('\n')

