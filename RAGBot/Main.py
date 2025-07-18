import os
if os.environ["HUGGINGFACEHUB_API_TOKEN"]:
    print("Huggingface token is set")
else:
    print("Huggingface token is not set")
    token = input("Enter your Huggingface token: ")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader, TextLoader, WebBaseLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
def data_loader(file):
    data = []
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file)
        data.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(file)
        data.extend(loader.load())
    elif file.endswith(".csv"):
        loader = CSVLoader(file)
        data.extend(loader.load())
    elif file.startswith("http" or "https"):
        loader = WebBaseLoader(file)
        data.extend(loader.load())
    else:
        loader = UnstructuredFileLoader(file)
        data.extend(loader.load())
    return data      
from langchain.text_splitter import RecursiveCharacterTextSplitter
def data_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    docs = text_splitter.split_documents(data)
    return docs
from langchain.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
def vector_database(docs):
    from langchain.vectorstores import Chroma
    vectordb = Chroma.from_documents(docs, embedding_model)
    return vectordb
def retriver(files):
    spliting = data_loader(files)
    chunks = data_splitter(spliting)
    vector_db = vector_database(chunks)
    retriver = vector_db.as_retriever()
    return retriver
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def chatbot(file, query):
    # Correctly wrap the HF pipeline
    hf_pipe = pipeline(
        task="text-generation",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        temperature=0.5,
        do_sample=False,
        device_map='auto',
        max_new_tokens=200,
        return_full_text=False,  # Optional, cleaner output
    )

    # Now wrap the pipeline correctly for LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the context below to answer the question concisely.

Context: {context}

Question: {question}

Answer:"""
    )

    # Your retriever logic
    retriever_obj = retriver(file)

    # Proper LLM object passed here
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj, chain_type_kwargs={"prompt": prompt})

    response = qa.run(query)
    result = re.sub(r'[^\x20-\x7E]+', ' ', response)
    return result

import gradio as gr
def chat_interface(user_message, chat_history, uploaded_file):
    if uploaded_file is None:
        bot_reply = "Please upload a file before chatting."
    else:
        try:
            bot_reply = chatbot(uploaded_file, user_message)
        except Exception as e:
            bot_reply = f"Error: {str(e)}"

    chat_history = chat_history or []
    chat_history.append((user_message, bot_reply))
    return chat_history, chat_history, ""  # Third return clears the textbox

with gr.Blocks() as demo:
    uploaded_file = gr.File(label="Upload your file")
    
    gr.Markdown("### Chatbot")

    chatbot_ui = gr.Chatbot()
    user_input = gr.Textbox(
        show_label=False,
        placeholder="Type your message and press Enter...",
        lines=1
    )

    user_input.submit(
        chat_interface,
        inputs=[user_input, chatbot_ui, uploaded_file],
        outputs=[chatbot_ui, chatbot_ui, user_input]  # Clear textbox here
    )
demo.launch(debug=True)


