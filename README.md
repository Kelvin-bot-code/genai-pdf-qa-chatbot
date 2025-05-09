## Development of a PDF-Based Question-Answering Chatbot Using LangChain

#### register no: 212224040157
#### name       : kelvin k


### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
The objective is to create a chatbot that can intelligently respond to queries based on information extracted from a PDF document. By using LangChain, the chatbot will be able to process the content of the PDF and use a language model to provide relevant answers to user queries. The effectiveness of the chatbot will be evaluated by testing it with various questions related to the document.
### DESIGN STEPS:

### Step 1: 
#### Loads the OpenAI API key from a .env file using dotenv.

### Step 2:
#### Uses PyPDFLoader to extract text from the given PDF document.

### Step 3:
#### Breaks the PDF text into overlapping chunks using RecursiveCharacterTextSplitter for better context handling.

### Step 4: 
#### Creates vector embeddings for each chunk using OpenAIEmbeddings and stores them in an in-memory vector database.

### Step 5: 
#### Uses LangChain’s ConversationalRetrievalChain with memory and a retriever to enable context-aware conversation.

### Step 6: 
#### Runs a command-line chatbot that answers user queries based on the content of the uploaded PDF.


### PROGRAM:
```py
import os
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
#loading data
def load_pdf_to_retriever(file_path):
    """Loads a PDF file and prepares a retriever using embeddings."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    return vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#building conversation
def build_conversational_chain(retriever):
    """Creates a conversational retrieval chain with memory."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

#starting chat
def start_chat(chatbot):
    """Starts a command-line chatbot interface."""
    print("Welcome to the PDF Question-Answering Chatbot!")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            print("Chatbot: Thanks for chatting! Goodbye!")
            break
        response = chatbot({"question": user_input})
        print("Chatbot:", response.get("answer", "Sorry, I couldn't process that."))

#function call        
if __name__ == "__main__":
    pdf_file_path = "docs/cs229_lectures/MachineLearning-Lecture01.pdf"  # Update path as needed
    retriever = load_pdf_to_retriever(pdf_file_path)
    chatbot = build_conversational_chain(retriever)
    start_chat(chatbot)
```
### OUTPUT:

![Screenshot 2025-05-10 011547](https://github.com/user-attachments/assets/164db2b6-4a83-4f31-bb8f-c9e36b2d9d84)

### RESULT:
The PDF Question-Answering Chatbot successfully processes PDFs, answers user queries with relevant information, and maintains conversation context.
