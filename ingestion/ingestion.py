import os
from dotenv import load_dotenv
import pandas as pd

import random
import humanize

from elasticsearch import Elasticsearch
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# env vars
load_dotenv()

doc_fname = os.getenv("DOC_FNAME", "data/state_of_the_union.txt")
es_cert_path = os.getenv("ES_CERT_PATH", "cert/rag-union-es-cert")

es_url = os.getenv("ES_URL", None)
es_user = os.getenv("ES_USER", None)
es_password = os.getenv("ES_PASSWORD", None)

index_name = os.getenv("INDEX_NAME", None)

# functions

## Function to load document file and split into texts
def load_and_split(fname):
    loader = TextLoader(fname)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("doc load and split complete")
    return texts

def output_index_stats(es):
    print("Index info")
    print("----")
    print(dict(es.indices.get(index=index_name)))
    print("  ")
    
    doc_count = es.count(index=index_name)["count"]
    print("Num of docs: " + str(doc_count))
    print("  ")

    # print random document as a sample
    print("Sample doc")
    print("----")
    print(dict(es.get(index=index_name, id=random.randint(0, len(documents)-1))))
    print("  ")

    index_stats = es.indices.stats(index=index_name).get('_all').get('primaries')
    print("Index size:    " + humanize.naturalsize(index_stats.get('store').get('size_in_bytes')))
    print("Indexing time: " + humanize.precisedelta(index_stats.get('indexing').get('index_time_in_millis')/1000, minimum_unit='minutes'))

# main
if __name__ == "__main__":

    # 1. load data 
    # texts = type 'list'
    # each item = 'langchain_core.documents.base.Document'
    texts = load_and_split(doc_fname)  

    # 2. create a dataframe for the set of Documents
    page_contents = [text.page_content for text in texts] # array of 'str'
    sources = [text.metadata['source'] for text in texts] # array of 'str'
    documents = pd.DataFrame({
        'page_content': page_contents,
        'source': sources
    })

    # 3. create an index for Elasticsearch
    index_name=index_name

    # 4. embedding function
    transformer_name="all-MiniLM-L6-v2"
    emb_func = SentenceTransformerEmbeddings(model_name=transformer_name)

    # 5. connect to Elasticsearch
    es = Elasticsearch(
        es_url,
        ca_certs=es_cert_path,
        basic_auth=(es_user, es_password)
    )
    print("connection object to Elasticsearch ready to use")

    # 6. index documents to Elasticsearch
    kb = ElasticsearchStore(
        es_connection=es,
        index_name=index_name,
        embedding=emb_func,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
        distance_strategy="DOT_PRODUCT"
    )
    print("create vector store in Elasticsearch")

    # Test the connection
    if es.ping():
        print("Connected successfully.")
    else:
        print("Connection failed.")

    try:
        _ = kb.add_texts(
            texts=documents['page_content'].tolist(),
            metadatas=[{'source': source} for source in documents['source']],
            index_name=index_name,
            ids=[str(i) for i in range(len(documents))]  # unique for each doc
        )
    except Exception as e:
        print("Failed to index documents:", e)

    # 7. stats
    ## this ingestion program is an update of texts into ES, not additional each time it runs
    ## you can see that in Number of Docs output after repeated runs
    ## this index stats output also retrieves a random document and you can see its version number 
    ## going up after each ingetion run indicate an update rather than an add
    output_index_stats(es)

    # LOADING COMPLETE
    print("loading complete")