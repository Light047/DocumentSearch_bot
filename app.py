import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# Function for URL processing
def process_input(urls, question):
    model_local = Ollama(model="mistral")  # instance for our ollama model
    
    # Convert strings of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]

    # Splitting documents into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encode(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    # Convert text chunks into embeddings and store in VB
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="RAG_chroma", embedding=embeddings.ollama.OllamaEmbeddings(model="nomic-embed-text"))
    retriever = vectorstore.as_retriever()

    # Performing Augmentation by using those Retrieval
    after_rag_template = """Answer the following question based on the following text."""
    # Prompt template for model
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"Context": retriever, "Question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke({"Context": retriever.retrieve(question), "Question": question})

# Now for Streamlit UI
st.title("Document Query by MISTRAL")
st.write("Enter URLs (one per line) and a question to query the document.")

# UI for input fields
urls = st.text_area("Enter URLs separated by lines:", height=150)
question = st.text_input("Question")

# Button to process input
if st.button('Query documents'):
    with st.spinner('Processing...'):
        answer = process_input(urls, question)
        st.text_area("Answer", value=answer, height=300, disabled=True)
