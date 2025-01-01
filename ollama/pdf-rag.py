import chromadb
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import (OnlinePDFLoader,
                                                  UnstructuredPDFLoader)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama

DOC_PATH = './data/BOI.pdf'
MODEL = 'llama3.2:latest'
MODEL_EMBEDDINGS = 'nomic-embed-text'

if DOC_PATH:
    loader = UnstructuredPDFLoader(file_path=DOC_PATH)
    data = loader.load()
    print('done loading a pdf')

else:
    print('upload a pdf')


# content = data[0].page_content
# print(content[:100])

# split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)

print('done splitting text')
# print(f"Total chunks: {len(chunks)}")
# print(f"First chunk: {chunks[0]}")


# pull embeddings from ollama.ai
ollama.pull(MODEL_EMBEDDINGS)

embeddings = OllamaEmbeddings(model=MODEL_EMBEDDINGS)

# embed = OllamaEmbeddings(
#     model="nomic-embed-text",
# )


chroma_client = chromadb.Client()
# collection = chroma_client.get_or_create_collection('simple-rag')

vector_db = Chroma(
    collection_name="simple-rag",
    embedding_function=embeddings,
    client=chroma_client,
)

vector_db.add_documents(chunks)

# vector_db = Chroma(
#     documents=chunks,
#     embeddings=embeddings,
#     collection_name="simple-rag",
# )

print('done adding to vector db')


# retrieval
llm = ChatOllama(model=MODEL)

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

# res = chain.invoke(input=("What is the document about?"))
res = chain.invoke(input=("how to report BOI?",))

print(res)
