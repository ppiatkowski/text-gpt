"""Ask a question about the ingested text"""
import faiss
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import argparse

parser = argparse.ArgumentParser(description='Ask a question to the book.')
parser.add_argument('question', type=str, help='The question to ask the book.')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
llm = OpenAI(temperature=0, max_tokens=256)
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=store.as_retriever()
)
result = chain({"question": args.question})
print(result['answer'])
# print(f"Sources: {result['sources']}")
