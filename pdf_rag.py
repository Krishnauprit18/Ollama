# Ingest the pdf files
# Extract text from the pdf files and split into small chunks
# send the chunks to the embedding model
# Save the embeddings to a vector database
# perform similarity search to the vector database to find similar documents
# retrieve the similar documents and present them to the user

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "/home/krishna/Documents/Ollama/data/Unit-7 Color IP.pdf"
model = "llama3.2"

# local pdf files upload:
if doc_path:
    loader = UnstructuredPDFLoader(file_path = doc_path)
    data = loader.load()
    print("Done Loading...")
else:
    print("Upload a PDF File")

content = data[0].page_content
print(content[:100]) 

# Extract text from PDF and split into smaller chunks:
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Split and chunk:
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
chunks = text_splitter.split_documents(data)
print("Done Splitting...")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")
 
 # Add to Vector Database:
import ollama
ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
         documents = chunks,
         embedding = OllamaEmbeddings(model = "nomic-embed-text"),
         collection_name = "simple_rag",
)
print("Done adding to vector database...")

# Retrieval:
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up our model:
llm = ChatOllama(model = model)

# a simple technique to generate multiple questions from a single question and retrieve document 
# based on those questions , getting the best of both worlds.
QUERY_PROMPT = PromptTemplate(
        input_variables = ["question"],
        template = """You are an AI language model assistent. Your task is to generate five
        different versions of the given user question to retrieve relevent documents from 
        a vector database. By generating multiple perspectives on the user question, your 
        goal is to help the user overcome some of the limitations of distance-based
        similarity search. provide these alternative questions seperated by newlines.
        Original Question: {question}""",
    )

retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt = QUERY_PROMPT
)

# RAG Prompt:
template = """Answer the questions based only on the following context:
    {context}
    Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


res = chain.invoke(input = ("What is the document about?",))
res1 = chain.invoke(input = ("what are the main points as an owner i should be aware of?"))
print(res)
print(res1)
    
