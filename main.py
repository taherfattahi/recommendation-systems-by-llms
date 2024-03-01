import pandas as pd
import tiktoken

import lancedb

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import LanceDB

openai_api_key = ""
client = OpenAI(api_key=openai_api_key)

anime = pd.read_csv('data/anime_with_synopsis.csv')
# anime.head()

anime['combined_info'] = anime.apply(
    lambda row: f"Title: {row['Name']}. Overview: {row['sypnopsis']} Genres: {row['Genres']}", axis=1)
# anime.head(2)

# print(anime)

embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

encoding = tiktoken.get_encoding(embedding_encoding)

# omit descriptions that are too long to embed
anime["n_tokens"] = anime.combined_info.apply(lambda x: len(encoding.encode(x)))
anime = anime[anime.n_tokens <= max_tokens]

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


anime["embedding"] = anime.combined_info.apply(lambda x: get_embedding(x, model=embedding_model))
# anime.head()

anime.rename(columns={'embedding': 'vector'}, inplace=True)
anime.rename(columns={'combined_info': 'text'}, inplace=True)
anime.to_pickle('data/anime.pkl')

uri = "dataset/sample-anime-lancedb"

db = lancedb.connect(uri)
table = db.create_table("anime", anime)

# embeddings = OpenAIEmbeddings(engine="text-embedding-ada-002")
embeddings = OpenAIEmbeddings(
    deployment="SL-document_embedder",
    model="text-embedding-ada-002",
    show_progress_bar=True,
    openai_api_key=openai_api_key)

docsearch = LanceDB(connection=table, embedding=embeddings)

# simple similarity computation
# query = "I'm looking for an animated action movie. What could you suggest to me?"
# docs = docsearch.similarity_search(query, k=1)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-1106",
    temperature=0,
    api_key=openai_api_key
)

# without prompt
# qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
#                                        return_source_documents=True)

# let’s say we are only interested in anime that, among their genres, are tagged as “Action”.
# df_filtered = anime[anime['Genres'].apply(lambda x: 'Action' in x)]
# qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
#                                        retriever=docsearch.as_retriever(search_kwargs={'data': df_filtered}),
#                                        return_source_documents=True)

# define custom prompt
template = """You are a movie recommender system that help users to find anime that match their preferences. 
Use the following pieces of context to answer the question at the end. 
For each question, suggest three anime, with a short description of the plot and the reason why the user migth like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Your response:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=docsearch.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)

query = "I'm looking for an action anime. What could you suggest to me?"

# Query and Response
with get_openai_callback() as cb:
    result = qa_chain({"query": query})

print(result['result'])
