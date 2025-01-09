import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import chromadb
from typing import List, Optional
from langchain.schema import Document

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

# Configuration class to hold constants


class Config:
    DOC_PATH = os.getenv('DOC_PATH', './data/BOI.pdf')
    MODEL = os.getenv('MODEL', 'phi4:latest')
    MODEL_EMBEDDINGS = os.getenv('MODEL_EMBEDDINGS', 'nomic-embed-text')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'simple-rag')


# Initialize logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_pdf(DOC_PATH: str) -> Optional[List[Document]]:
    """
    Loads a PDF from the specified path and returns a list of Document objects.
    """
    try:
        loader = UnstructuredPDFLoader(file_path=DOC_PATH)
        data = loader.load()
        logging.info('Done loading a PDF')
        return data
    except FileNotFoundError:
        logging.error(f"PDF file not found at path: {DOC_PATH}")
        return None
    except Exception as e:
        logging.error(f"Error while loading PDF: {e}")
        return None


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks based on the specified chunk size and overlap.
    """
    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    logging.info('done splitting documents')
    return chunks


def initialize_embeddings(model_name: str) -> OllamaEmbeddings:
    """
    Initializes and returns the OllamaEmbeddings model.
    """
    ollama.pull(model_name)
    return OllamaEmbeddings(model=model_name)


# @st.cache_resource
def create_vector_db(_chunks: List[Document], embeddings: OllamaEmbeddings, collection_name: str) -> Chroma:
    """
    Creates a vector database, adds document chunks, and returns the database object.
    """
    chroma_client = chromadb.Client()
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=chroma_client,
    )
    vector_db.add_documents(_chunks)

    logging.info('done adding to vector db')
    return vector_db


def create_retriever(vector_db, llm):
    # retrieval
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info('done creating retriever')
    return retriever


def create_chain(retriever, llm):
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {
            "context": retriever, "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info('done creating chain')
    return chain


def main():
    st.title("Document Assistant")

    # user input
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                llm = ChatOllama(model=Config.MODEL)

                # Load the vector database
                data = load_pdf(Config.DOC_PATH)
                chunks = split_documents(data)
                embeddings = initialize_embeddings(Config.MODEL_EMBEDDINGS)
                vector_db = create_vector_db(
                    _chunks=chunks, embeddings=embeddings, collection_name=Config.COLLECTION_NAME)
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()
