#set up imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv


# get openai api key
load_dotenv()
key = os.environ.get("OPENAI_API_KEY")

#set up langchain language model
llm = ChatOpenAI(openai_api_key = key)
chat_model = ChatOpenAI()

# Chain imports
import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader

# # loads and store QnA txt file (information for QnA)
# loader = TextLoader("QnA.txt")
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    formatted_history = "\n\n".join(line for line in history)
    return formatted_history

# Chat GPT prompt to make it act as a customer service bot
template = """You are a customer service bot that is in training mode, try to answer questions with the context, but if unable to,
ask for the correct answer and then acknowledge the correct answer.
{context}
Question: {message}
Reply:"""
rag_prompt_custom = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# intial elements to store questions and history
conversation_history = []
question = ""

# update the RAG after more information is added into the QnA txt file
def update_rag():
    # reloads new updated file
    loader = TextLoader("QnA.txt")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    # create new chain
    rag_chain = (
    {"context": retriever | format_docs,  "message":RunnablePassthrough()}
    | rag_prompt_custom
    | llm
    | StrOutputParser()
    )
    return rag_chain

# adds question answer pair to data when an answer is made
def add_to_data(question, answer):
    with open("QnA.txt", "a") as f:
        f.write("\n")
        f.write("Q: " + question + "\n")
        f.write(answer + "\n")
        f.close()

    rag_chain = update_rag()

# set up retrival augmentmented Generation chain (retrieves answer to question from knowledge base)
rag_chain = update_rag()

# def chatter():
#     while input != "quit()":
#         message = input("Customer: ")
#         conversation_history.append(message)
#         if message.startswith("A: "):
#             add_to_data(conversation_history[-3], conversation_history[-1])
#         reply = rag_chain.invoke(message)
#         conversation_history.append(reply)


#        print(reply)
        
# Flask to create API
from flask import Flask, request, render_template, make_response, flash

app = Flask(__name__) 
app.secret_key = "secretkey"
# Set up of orginal page
@app.route('/')
def my_form():
    global question
    return render_template('my-form.html')

# Webpage update after sending a message
@app.route('/', methods=['POST'])
def my_form_post():
    global question
    rag_chain = update_rag()
    message = request.form['text']
    if not message.startswith("A: "):
        question = message
    if message.startswith("A: "):
        print(question)
        add_to_data(question, message)
        rag_chain = update_rag()
    conversation_history.append(message)
    
    reply = rag_chain.invoke(message)
    conversation_history.append(reply)
    outtext = "break".join(conversation_history)
    response = make_response(render_template('my-form.html', text=outtext))
    return response

if __name__ == '__main__':
    app.run(debug=True, port=8000)