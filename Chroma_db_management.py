
# this software is mainly derivated from the example found at the following link https://github.com/rubentak/Langchain/blob/main/notebooks/Langchain_doc_chroma.ipynb

print("Load DB")
import time
start_time = time.time()

#streamlit bug fix https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    print("pysqlite not available")

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import uuid
from streamlit import secrets
print(f"Time to load DB main lib: {time.time() - start_time}")
#import Raptor as Raptor
print(f"Time to load Raptor: {time.time() - start_time}")

API_KEY = secrets['openai_api_key'] # open ai api key shall be created in openai website
EMBEDDING_MODEL = OpenAIEmbeddings(openai_api_key = API_KEY, model = "text-embedding-3-large")

class ChromaDb(object):
    def __init__(self, txt_directory = './Database/Public/Files/', chroma_db_directory = './Database/Public/Chroma_DB/'):
        
        # init some variables
        print("Start to load a database")
        start_time = time.time()
        self.unstructured_available = False
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
            from langchain_community.document_loaders.directory import DirectoryLoader # need to be connected to GUEST cause nltk could need to download external data
            from langchain_unstructured import UnstructuredLoader
            from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
            self.DirectoryLoader = DirectoryLoader
            self.UnstructuredLoader = UnstructuredLoader
            self.unstructured_available = True
            self.UnstructuredPDFLoader = UnstructuredPDFLoader
            print("Unstructured loaded / DB can be edited")
        except ImportError:
            print("Unstructured library not available. Editing is not possible")

        self.api_key = API_KEY
        self.txt_directory = txt_directory # use '\\' instead of '/' cause chroma db replaces all '/' which leads some function like delete docu to not work properly
        self.chroma_db_directory = chroma_db_directory
        self.chunk_size=1000    #this value shall be constant for a database. If it is modified for an already existing database, it will lead to duplicates in the database
        self.chunk_overlap=200  #this value shall be constant for a database. If it is modified for an already existing database, it will lead to duplicates in the database
        self.embedding = EMBEDDING_MODEL
        self.summaries = ""
        self.results = ""
        self.vectordb = Chroma(persist_directory=self.chroma_db_directory,
              embedding_function=self.embedding)
        print(f"Time to load a DB: {time.time() - start_time}")
    
    # goal is to loop the DB and get a list with unique source seen
    def list_all_chromadb_files(self):
        files = self.vectordb.get()['metadatas']
        
        seen_sources = set()
        unique_data = []

        for item in files:
            if item['source'] not in seen_sources:
                #st.write(item['source'])
                unique_data.append(item)
                filename = item['source'].replace('/', '\\').rsplit('\\', 1)[1]
                seen_sources.add(filename)
        seen_sources = list(seen_sources)
        #print(seen_sources)
        return seen_sources
    
    # simple wrapper function if a list of text is needed
    def return_documents_as_a_list(self):
        return list(self.vectordb.get()['documents'])
    
    # this function load all the document in the folder txt_directory, 
    # it will perform a id verification, meaning that if a newest version is added, it will added to the database too
    def add_a_directory_to_chroma_database(self, txt_directory, *args):
        # this function will load all the content of folder into a list. 
        # A good example can be found at https://python.langchain.com/docs/integrations/document_loaders/unstructured_file.html
        if self.unstructured_available == False:
            print("Chroma editing / Unstructured library not available.")
            return None
        loader = self.DirectoryLoader(path = txt_directory, show_progress=True, use_multithreading = True)
        self.add_a_loader_to_database(loader = loader, *args)

    
    # this function will add all missing document to the database, aim is to be faster than loaded the whole directory
    def add_missing_document_to_chroma_database(self, *args):
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
            filename_without_path = filename.rsplit('\\', 1)[1] # keep filename and remove the rest. Split name with delimeter and save the result in a 2 items array
            if filename_without_path in filenames_database:
                print(f"filename {filename_without_path} already in the database")
            else:
                print(f"filename {filename_without_path} not in the database")
                self.add_or_delete_a_document(*args, filepath = filename, method = 'Add')

    # do no forget the method
    def add_or_delete_a_document(self, *args, filepath, method=None ):
        # add method
        if method == 'Add':
            if self.unstructured_available == False:
                print("Chroma editing / Unstructured library not available.")
                return None
            loader = self.UnstructuredLoader(filepath)
            self.add_a_loader_to_database(*args, loader = loader)
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

    # simple function to 
    def add_a_loader_to_database(self, *args, loader):
        document = loader.load()
        self.add_a_document_to_database(*args, document = document)

    # function to pre process document using RAG Raptor
    def process_document_using_Raptor(self, texts):
        leaf_texts = [doc.page_content for doc in texts]
        self.results = Raptor.recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
        # Initialize all_texts with leaf_texts
        all_texts = leaf_texts.copy()

        # Iterate through the results to extract summaries from each level and add them to all_texts
        for level in sorted(self.results.keys()):
            # Extract summaries from the current level's DataFrame
            self.summaries = self.results[level][1]["summaries"].tolist()
            # Extend all_texts with the summaries from the current level
            all_texts.extend(self.summaries)
            text = Document(page_content = self.summaries[0], metadata = texts[0].metadata)
            texts.append(text)

        return texts

    # a loader can be a directory, a single file or whatever ...
    # this function will check if the loader (or the file) is already in the database and it only is it is missing, avoid duplicates
    def add_a_document_to_database(self, *args, document):
        # Step 1: Remove items where metadata['category'] is 'header' or 'footer'
        filtered_documents = [doc for doc in document if doc.metadata['category'] not in ['header', 'footer']]

        # Step 2: Identify duplicates based on the first line of doc.metadata['page_content']
        seen_content = set()  # Set to keep track of seen page contents

        # Loop through filtered documents and remove duplicates
        unique_documents = []
        for doc in filtered_documents:
            first_line = doc.page_content  # Get the first line of the content
            if first_line not in seen_content:
                unique_documents.append(doc)  # Add to unique list if not a duplicate
                seen_content.add(first_line)  # Mark this content as seen
                
        document = unique_documents
        
        # Splitting the text into chunks
        # increase efficiency of embeddings searching function
        # some explanation can be found here https://dev.to/peterabel/what-chunk-size-and-chunk-overlap-should-you-use-4338
        # overlap is to avoid to split data in a middle on an important sentence
        text_splitter = self.RecursiveCharacterTextSplitter (chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(document)
        
        #RAPTOR
        for arg in args:
            print(f"arg {arg} requested")
            if arg == 'Raptor':
                texts = self.process_document_using_Raptor(texts)
                #for text, leaf_text in zip(texts, leaf_texts):
                #    text.page_content = leaf_text
        
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
                                     collection_metadata={"hnsw:space": "l2"}) # following the link below, l2 seems better than cosine as distance metrics https://medium.com/@stepkurniawan/comparing-similarity-searches-distance-metrics-in-vector-stores-rag-model-f0b3f7532d6f

            # Save the db locally to disk
            self.vectordb.persist()

    
        
if __name__ == "__main__":
    print("done")
    #document = UnstructuredLoader(file_path=path, chunking_strategy="basic", strategy="auto",).load()
    
