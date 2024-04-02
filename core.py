from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
import sys

######################## SEARCH & EXTRACT DOCUMENT ################################

def input_document():
    document=[]
    for file in os.listdir("uploads"):
        if file.endswith(".pdf"):
            pdf_path="./uploads/"+file
            loader=PyPDFLoader(pdf_path)
            document.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path="./uploads/"+file
            loader=Docx2txtLoader(doc_path)
            document.extend(loader.load())
        elif file.endswith('.txt'):
            text_path="./uploads/"+file
            loader=TextLoader(text_path)
            document.extend(loader.load())

    print("Length of the document : ",len(document))

    return document

# df_document = input_document()

######################## Split Document into Chunks ################################

def split_document(document):

    document_splitter=CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
    document_chunks=document_splitter.split_documents(document)
    print("Length of document chunck : ", len(document_chunks))

    return document_chunks

# df_chunks = split_document(df_document)

######################## Download the Embeddings from Hugging Face, Download the Sentence  ################################
def init_huggingface(API_KEY):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    os.environ["OPENAI_API_KEY"]= API_KEY
    embeddings = OpenAIEmbeddings()
    notebook_login()

    return embeddings

# embedding = init_huggingface("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")

######################## Setting Up Chroma as our Vector Database  ################################
def init_chroma(chunks,embedding):
    vectordb=Chroma.from_documents(chunks,embedding=embedding, persist_directory='./data')
    vectordb.persist()

    return vectordb

# db_vector = init_chroma(df_chunks, embedding)


######################## Download Lamma Model  ################################

def setup_model(dbvector):

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                            use_auth_token=True,)


    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                use_auth_token=True,
                                                )
    pipe=pipeline("text-generation",
              model=model,
              tokenizer=tokenizer,
              torch_dtype=torch.bfloat16,
              device_map='auto',
              max_new_tokens=512,
              min_new_tokens=-1,
              top_k=30
              )
    llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})
    llm=ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')

    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    pdf_qa=ConversationalRetrievalChain.from_llm(llm=llm,
                                             retriever=dbvector.as_retriever(search_kwargs={'k':6}),
                                             verbose=False, memory=memory)

    return pdf_qa

# pdf_model = setup_model(db_vector)

######################## Creating a Conversation Retrieval QA Chain ################################


def start_conversation_prompt(model):
    print('---------------------------------------------------------------------------------')
    print('-----------Start The Conversation With Document Based Knowledge------------------')
    print('---------------------------------------------------------------------------------')

    while True:
        query=input(f"Prompt:")
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
            sys.exit()
        if query == '':
            continue
        result = model({"question": query})
        print(f"Answer: " + result["answer"])
        return result

def start_conversation(model, question):
    print("Question :", question)
    result = model({"question":question})
    return result

# start_conversation(pdf_model)
