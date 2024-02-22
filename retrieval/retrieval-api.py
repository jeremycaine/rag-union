import flask
import os
from dotenv import load_dotenv

from elasticsearch import Elasticsearch
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import SentenceTransformerEmbeddings

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

# for BAM
#from genai.model import Model

# for WML
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from langchain.chains import RetrievalQA

load_dotenv()

es_url = os.getenv("ES_URL", None)
es_user = os.getenv("ES_USER", None)
es_password = os.getenv("ES_PASSWORD", None)
index_name = os.getenv("INDEX_NAME", None)
es_cert_path = os.getenv("ES_CERT_PATH", "cert/rag-union-es-cert")

model_id_name = os.getenv("MODEL_ID_NAME", None)
wml_project_id = os.getenv("WML_PROJECT_ID", None)

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

# RETRIEVAL

# 1. embedding function
transformer_name="all-MiniLM-L6-v2"
emb_func = SentenceTransformerEmbeddings(model_name=transformer_name)

# 2. connection object to elasticsearch
es = Elasticsearch(
    es_url,
    ca_certs=es_cert_path,
    basic_auth=(es_user, es_password)
)
print("connection object to Elasticsearch ready to use")

# 3. connection to vector store
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

# 4. create retriever objects
retriever = kb.as_retriever(embedding_function=emb_func)
print(retriever)

# 5. setup model
model_id    = model_id_name
gen_parms   = None
project_id = wml_project_id
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

watsonx_granite = WatsonxLLM(model=model)
print(watsonx_granite)

# 6. set up query engine
qa = RetrievalQA.from_chain_type(
    llm=watsonx_granite,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# web app api endpoint
app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "ready"

@app.route('/query', methods=['POST'])
def image():
    data=flask.request.get_json()
    question = data.get('query', None)

    # question = 'What did the president say about Ketanji Brown Jackson'
    # question = 'what did the president say about powerful economic sanctions'
    # question = 'how many barrells of Strategic Petroleum Reserve'

    result = qa({"query": question})
    response = result['result']
    print(response)
    return flask.jsonify({'response': f'{response}'})


# Get the PORT from environment
port = os.getenv('PORT', '8081')
debug = os.getenv('DEBUG', 'false')
if __name__ == "__main__":
    print("application ready - Debug is " + str(debug))
    app.run(host='0.0.0.0', port=int(port), debug=debug)
