"""Ingesting text data into LangChain."""
import argparse
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

parser = argparse.ArgumentParser(description='Ingest a text file')
parser.add_argument('path', type=str, help='Path to a text file to ingest')
args = parser.parse_args()

data = []
with open(args.path) as f:
    data.append(f.read())

# We need to split the text due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": 'text'}] * len(splits))

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
