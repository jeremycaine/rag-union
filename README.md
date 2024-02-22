# State of the Union RAG
Chat interface to RAG application with corpus of the State of the Union speech using Elasticsearch from IBM Cloud Databases and LLMs hosted on IBM watsonx.ai service.

## 0. Environment Setup

### IBM Cloud
#### Elastic
Create an instance of
- IBM Databases for Elasticsearch

In the Elasticsearch instance dashbaord, go to Service Credentials, and Create new credentials, returned as a JSON object.

`ES_URL` is `connection.https.composed`
`ES_USER` is `connection.https.authentication.username`
`ES_PASSWORD` is `connection.https.authentication.password`

#### watsonx.ai
Create instances of
- IBM Watson Studio
- IBM Watson Machine Learing (WML)
and associate the WML instance in Studio | your project | Manage | Service Integrations.

## 1. Project Environment
In project directory, setup env vars in `.env`

### Ingestion
```
ES_URL=https://xxx.xxx.databases.appdomain.cloud:nnnnn
ES_USER=xxx
ES_PASSWORD=xxx
ES_CERT_PATH=cert/rag-union-es-cert
DOC_FNAME=data/state_of_the_union.txt
INDEX_NAME=union_speech
```
### Retrieval
Option to change model name and ES index name. Make sure you have the right WML project id.
```
ML_URL= use URL associated with region Studio is deployed e.g. Dallas is https://us-south.ml.cloud.ibm.com
ML_KEY= your IBM Cloud Key
ES_URL=https://xxx.xxx.databases.appdomain.cloud:nnnnn
ES_USER=xxx
ES_PASSWORD=xxx
ES_CERT_PATH=cert/rag-union-es-cert
MODEL_ID_NAME=ibm/granite-13b-chat-v1
WML_PROJECT_ID=xxxx
INDEX_NAME=union_speech
```
### Python environments

Create a Python 3.11 environment
```
conda create -n rag-union python=3.11
conda activate rag-union
pip install -r requirements.txt
```
### Elasticsearch certificate
Create a certificate file [`cert/rag-union-es-cert`](data/rag-union-es-cert)

In the Elasticsearch instance dashbaord, get the certificate contents from Overview | Endpoints | HTTPS | TLS certificate.

### Source data
The State of the Union speech text is in [`data/state_of_the_union.txt`](data/state_of_the_union.txt)

## 2. Ingestion
```
cd ingestion
pip install -r requirements.txt
python ingestion.py
```

## 3. Retrieval
```
cd retrieval
pip install -r requirements.txt
python retrieval-api.py
```
Call API as per 
```
curl ...
```

## 4. Testing
Healthcheck
```
curl http://127.0.0.1:8081
> ready
```
Sample questions
- question = 'What did the president say about Ketanji Brown Jackson'
- question = 'what did the president say about powerful economic sanctions'
- question = 'how many barrells of Strategic Petroleum Reserve'
```
curl -X POST http://127.0.0.1:8081/query \
-H "Content-Type: application/json" \
-d '{"query": "how many barrells of Strategic Petroleum Reserve"}'
```



