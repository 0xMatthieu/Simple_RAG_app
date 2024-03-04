
# this software is mainly derivated from the example found at the following link https://github.com/rubentak/Langchain/blob/main/notebooks/Langchain_doc_chroma.ipynb

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import uuid
import streamlit as st

#streamlit bug fix
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    print("pysqlite not available")

class ChromaDb(object):
    def __init__(self, txt_directory = 'Database\\Public\\Files', chroma_db_directory = 'Database\\Public\\Chroma_DB\\'):
        
        # init some variables
        self.unstructured_available = False
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
            from langchain_community.document_loaders.directory import DirectoryLoader, UnstructuredFileLoader # need to be connected to GUEST cause nltk could need to download external data
            self.DirectoryLoader = DirectoryLoader
            self.UnstructuredFileLoader = UnstructuredFileLoader
            self.unstructured_available = True
            print("Unstructured loaded / DB can be edited")
        except ImportError:
            print("Unstructured library not available. Editing is not possible")

        self.api_key = st.secrets['openai_api_key'] # open ai api key shall be created in openai website
        self.txt_directory = txt_directory # use '\\' instead of '/' cause chroma db replaces all '/' which leads some function like delete docu to not work properly
        self.chroma_db_directory = chroma_db_directory
        self.chunk_size=1000    #this value shall be constant for a database. If it is modified for an already existing database, it will lead to duplicates in the database
        self.chunk_overlap=200  #this value shall be constant for a database. If it is modified for an already existing database, it will lead to duplicates in the database
        self.embedding = OpenAIEmbeddings(openai_api_key = self.api_key, model = "text-embedding-3-large")
        self.vectordb = Chroma(persist_directory=self.chroma_db_directory,
              embedding_function=self.embedding)
    
    # goal is to loop the DB and get a list with unique source seen
    def list_all_chromadb_files(self):
        files = self.vectordb.get()['metadatas']
        
        seen_sources = set()
        unique_data = []

        for item in files:
            if item['source'] not in seen_sources:
                unique_data.append(item)
                seen_sources.add(item['source'].rsplit('\\', 1)[1])
        seen_sources = list(seen_sources)
        #print(seen_sources)
        return seen_sources
    
    # this function load all the document in the folder txt_directory, 
    # it will perform a id verification, meaning that if a newest version is added, it will added to the database too
    def add_a_directory_to_chroma_database(self, txt_directory):
        # this function will load all the content of folder into a list. 
        # A good example can be found at https://python.langchain.com/docs/integrations/document_loaders/unstructured_file.html
        if self.unstructured_available == False:
            print("Chroma editing / Unstructured library not available.")
            return None
        loader = self.DirectoryLoader(path = txt_directory, show_progress=True, use_multithreading = True)
        self.add_a_loader_to_database(loader = loader)

    
    # this function will add all missing document to the database, aim is to be faster than loader the whole directory
    def add_missing_document_to_chroma_database(self):
        # this function will load all the content of folder into a list. 
        # A good example can be found at https://python.langchain.com/docs/integrations/document_loaders/unstructured_file.html
        
        # get a list of all files in the directory
        filenames_directory = os.listdir(self.txt_directory)

        # get a list of all files in the vector database
        filenames_database = set(self.list_all_chromadb_files())
        
        # loop directory filename and check if name exists in database
        for filename in filenames_directory:
            #add folder name
            filename = self.txt_directory.replace('/', '\\') + filename
            filename = filename.rsplit('\\', 1)[1] # keep filename and remove the rest. Split name with delimeter and save the result in a 2 items array
            if filename in filenames_database:
                print(f"filename {filename} already in the database")
            else:
                print(f"filename {filename} not in the database")
                self.add_or_delete_a_document(filepath = filename, method = 'Add')

    # do no forget the method
    def add_or_delete_a_document(self, filepath, method = None):
        # add method
        if method == 'Add':
            if self.unstructured_available == False:
                print("Chroma editing / Unstructured library not available.")
                return None
            loader = self.UnstructuredFileLoader(filepath)
            self.add_a_loader_to_database(loader = loader)
        # delete method
        elif method == 'Delete':
            
            # get current data
            data = self.vectordb.get()
            
            # get source of current database
            seen_source = list(data.get('metadatas'))
            ids_to_delete = []

            for idx in range(len(data['ids'])):
                id = data['ids'][idx]
                metadata = data['metadatas'][idx]
                if metadata['source'] == filepath:
                    ids_to_delete.append(id)

            if len(ids_to_delete) > 0:
                print("document has been found and deleted")
                self.vectordb.delete(ids=ids_to_delete)
            else:
                print("no document found, nothing has been deleted")

            # Save the db locally to disk
            self.vectordb.persist()
   
        # else print a warning
        else:
            print("none or unknow method given")
    
    # a loader can be a directory, a single file or whatever ...
    # this function will check if the loader (or the file) is already in the database and it only is it is missing, avoid duplicates
    def add_a_loader_to_database(self, loader):
        docs = loader.load()
        # Splitting the text into chunks
        # increase efficiency of embeddings searching function
        # some explanation can be found here https://dev.to/peterabel/what-chunk-size-and-chunk-overlap-should-you-use-4338
        # overlap is to avoid to split data in a middle on an important sentence
        text_splitter = self.RecursiveCharacterTextSplitter (chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(docs)
        
        
        # generate unique id to add only missing texts
        # found on following link : https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, text.page_content)) for text in texts]
        unique_ids = list(set(ids))
        data = self.vectordb.get()
        
        # get ids of current database
        seen_ids = list(set(data.get('ids')))
        unique_texts = []
        unique_ids = []  # List to collect unique IDs
    
        # this line compares ids of text and database
        for text, id in zip(texts, ids):
            if id not in seen_ids:
                seen_ids.append(id)
                unique_texts.append(text)
                unique_ids.append(id)
        
        if len(unique_texts) > 0:
            print("document added to the DB")
            self.vectordb.add_documents(documents=unique_texts,
                                     ids=unique_ids,
                                     collection_metadata={"hnsw:space": "cosine"})

            # Save the db locally to disk
            self.vectordb.persist()

        
        
if __name__ == "__main__":
    print("done")
    
