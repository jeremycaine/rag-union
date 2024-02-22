import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

# for BAM
#from genai.model import Model

# for WML
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from langchain.chains import RetrievalQA
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()

host=os.environ.get("DB_HOST", None)
port=os.environ.get("DB_PORT", None)
database=os.environ.get("DB_NAME", None)
user=os.environ.get("DB_USER", None)
password=os.environ.get("DB_PASSWORD", None)
table_name = os.getenv("TBL_NAME", None)
es_url = os.getenv("ES_URL", None)
es_user = os.getenv("ES_USER", None)
es_password = os.getenv("ES_PASSWORD", None)

model = SentenceTransformer('all-MiniLM-L6-v2')

# functions
## get ML platform credentials
def get_creds():
    wml_url = os.environ.get("ML_URL", None)
    api_key = os.environ.get("ML_KEY", None)
    creds = {
        "url": wml_url,
        "apikey":api_key
    }
    print(creds)
    return creds


def set_parameters():
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 100
    }
    return parameters

class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
    
db_name = "vector_db"
host = "localhost"
password = "password"
port = "5432"
user = "acme"

# main
if __name__ == "__main__":

    # 1. connection to vector store
    conn = psycopg2.connect(
        dbname=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
    )
    conn.autocommit = True
    print("connect to db success")

    ## get vector store
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name="union_speech",
        embed_dim=384,  # openai embedding dimension
    )
    print("created connection to vector")

    # 2. setup model
    model_id    = "ibm/granite-13b-chat-v1"
    gen_parms   = None
    project_id = "a8e93470-8c18-4b93-b2a3-2c90ef1e5a8d"
    space_id    = None
    verify      = False

    creds = get_creds()
    params = set_parameters()
    print(params)

    model = Model(
        model_id=model_id, 
        params=params, 
        credentials=creds,
        project_id=project_id)

    model_instance = WatsonxLLM(model=model)
    print(model_instance)

    # 3. Generate a query embedding
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

    # 4. question
    # this would be REST API /query entry point

    # 'what did the president say about powerful economic sanctions',
    # "how many barrells of Strategic Petroleum Reserve"
    query_str = "What did the president say about Ketanji Brown Jackson"
    query_embedding = embed_model.get_query_embedding(query_str)

    # 5. Query the vector datbase
    # construct vector store query

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )

    # returns a VectorStoreQueryResult
    query_result = vector_store.query(vector_store_query)
#    print(query_result.nodes[0].get_content())
    print(query_result)

    # 6. Parse result into a set of nodes
    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))

    # 7. Put into a retriever
    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=2
    )
    print(retriever)

    # 8. query model
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=model_instance)
    result = query_engine.query(query_str)
    print(result)

# https://developer.ibm.com/blogs/awb-retrieval-augmented-generation-with-langchain-and-elastic-db
    
# change from Pg to host
    # split goog union and fire up ES, pattern in powerpoint, langchain