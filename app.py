
import streamlit as st
from streamlit_chat import message

import streamlit as st
import tempfile
import sys
import os
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

DIR_PATH = "./data"
# Setting page title and header
st.set_page_config(page_title="ChatwithDoc", page_icon="	:page_with_curl:")
st.markdown("<h1 style='text-align: center;'>Chat with your Docs</h1>", unsafe_allow_html=True)

def clear_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                clear_directory(item_path)
                os.rmdir(item_path)
    else:
        st.warning(f'The directory with path {directory_path} doesnt exist', icon="⚠️")


# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'data_dir' not in st.session_state:
    clear_directory(DIR_PATH)
    st.session_state['data_dir'] = ""

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
uploaded_file = st.sidebar.file_uploader('Choose Files and Click Upload Files', type=['txt','pdf','doc','docx'], accept_multiple_files= True)
upload_button = st.sidebar.button("Upload Files", key="upload")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Conversation", key="clear")

def process_docs(uploads):
    documents = []
    for file in uploads:
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
    return documents


if upload_button:
    with st.spinner('Uploading...'):
        if uploaded_file is not None:
            documents = process_docs(uploaded_file)
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
        texts = text_splitter.split_documents(documents)
        st.session_state['texts'] = texts
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        st.session_state['embeddings'] = embeddings
    if embeddings:
        st.sidebar.write("Uploading done")
    

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = []
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    st.session_state['texts'] = []
    st.session_state['embeddings'] = []
    clear_directory(DIR_PATH)
    st.session_state['data_dir'] = ""



# generate a response
def generate_response(prompt, texts, embeddings):
    st.session_state['data_dir'] = DIR_PATH
    db = Chroma.from_documents(texts, embedding = embeddings,persist_directory = st.session_state['data_dir'])
    db.persist()
    # Create retriever interface
    retriever = db.as_retriever(search_kwargs={'k': 7})
    qa_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(openai_api_key=openai_api_key),
    retriever  = retriever,
    return_source_documents=True
    )
    response =  qa_chain({'question': prompt, 'chat_history': st.session_state['messages']})
    st.session_state['messages'].append((prompt, response['answer']))
    return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input, st.session_state['texts'], st.session_state['embeddings'])
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output['answer'])
        st.session_state['model_name'].append(model_name)
       
        

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
           