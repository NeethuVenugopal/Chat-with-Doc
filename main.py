import streamlit as st
import tempfile
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

def process_docs(uploads):
    documents = []
    for file in uploads:
        print(file)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            filename = file.name
            print(filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file.name)
                documents.extend(loader.load())
            elif filename.endswith('.docx') or filename.endswith('.doc'):
                loader = Docx2txtLoader(tmp_file.name)
                documents.extend(loader.load())
            elif filename.endswith('.txt'):
                loader = TextLoader(tmp_file.name)
                documents.extend(loader.load())
    print("...............")
    print(documents)
    return documents
def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        print(uploaded_file)
        documents = process_docs(uploaded_file)
        # documents = [uploaded_file.read().decode()]
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

# Page title
st.set_page_config(page_title='Chat with your Documents')
st.title('Chat with your Documents')

# File upload
uploaded_file = st.file_uploader('Upload an article', type=['txt','pdf','doc','docx'], accept_multiple_files= True)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)

