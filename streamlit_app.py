import os, tempfile
import pinecone
from pathlib import Path
from pinecone import ServerlessSpec
from pinecone import Index
from pinecone import Pinecone as PineconeClient

import openai
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_community.llms import OpenAIChat
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

import streamlit as st

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="Michael-RAG")
st.title("Enhanced Q&A")
st.write("Leveraging Pinecone, LangChain, and OpenAI for Generative Question Answering with Retrieval Augmented Generation (RAG)")


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def embeddings_on_pinecone(texts):

    # Initialize Pinecone client with the API key
    pc = PineconeClient(api_key=st.session_state.pinecone_api_key)

    # Set up embeddings using the OpenAI API key
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)

    # Create a Pinecone vector store from documents
    vectordb = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=st.session_state.pinecone_index
    )

    # Create a retriever
    retriever = vectordb.as_retriever()

    return retriever

def query_llm(retriever, query):
    # Create the conversational chain with the OpenAIChat model
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),  # Specify the model you want to use
        retriever=retriever,
        return_source_documents=True,
    )
    # Perform the query and handle the results
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})

    # Extract the answer and source documents
    answer = result['answer']
    source_documents = result['source_documents']  # Extract the source documents used

    # Append the query and answer to the session state
    st.session_state.messages.append((query, answer))

    return answer, source_documents

def input_fields():
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
        #
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else: 
            st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")
        #
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone environment")
        #
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone index name")
    #
    #st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
    #
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    #


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                #
                #if not st.session_state.pinecone_db:
                    #st.session_state.retriever = embeddings_on_local_vectordb(texts)
                #else:
                st.session_state.retriever = embeddings_on_pinecone(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    input_fields()
    st.button("Submit Documents", on_click=process_documents)

    # Initialize the messages session state if it does not exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize the retriever session state if it does not exist
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # Display the chat history
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    
    # Handle user query input
    if query := st.chat_input():
        st.chat_message("human").write(query)

        # Get the answer and source documents
        answer, source_documents = query_llm(st.session_state.retriever, query)
        
        # Combine the answer and formatted citations for AI's response
        response = f"**Answer:** {answer}\n\n**Sources Used:**\n"
        
        # To keep track of unique content and avoid duplicates
        unique_contents = set()
        formatted_sources = []
        
        # Collect unique content and format it
        for source in source_documents:
            if source.page_content not in unique_contents:
                unique_contents.add(source.page_content)
                formatted_content = format_source_content(source.page_content)
                formatted_sources.append(formatted_content)
        
        # Format the sources for display with bullet points and clear sections
        for i, content in enumerate(formatted_sources, start=1):
            response += f"{i}. {content}\n\n"

        # Write the combined response as the AI message
        st.chat_message("ai").write(response)

def format_source_content(content):
    """
    Formats the source content to make it more readable.
    Splits the content into sentences or bullet points where appropriate.
    """
    # Replace bullet points or other symbols to create structured lists
    formatted_content = content.replace("●", "\n-").replace("•", "\n-")

    # Add line breaks after periods for clearer separation of sentences
    formatted_content = formatted_content.replace(". ", ".\n")

    return formatted_content

if __name__ == '__main__':
    boot()