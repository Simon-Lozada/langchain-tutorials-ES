import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk

# Uncomment the following lines if needed
os.environ['OPENAI_API_KEY'] = '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'
# nltk.download('averaged_perceptron_tagger')

file = st.text_input("Insert the path to the file")
loader = DirectoryLoader(file, glob='.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch = Chroma.from_documents(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

query = st.text_input("Enter your query")
if query:
    result = qa.run(query)
    st.write(result)
