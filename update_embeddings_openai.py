import os

import openai
import pymongo

openai.api_key = os.environ['OPENAI_API_KEY']
embedder = openai.Embedding
client = pymongo.MongoClient(os.environ['MONGODB_URI'])
db = client.get_database('ai_ml_playground')
collection = db.get_collection('personnel')

personnel = collection.find()

print('***EMBEDDINGS UPDATE TOOL***')
print('Starting to update the embeddings for {} personnel'.format(collection.count_documents({})))
embeddings = []
for doc in personnel:
    raw = openai.Embedding.create(input=[doc.get('bio').replace('\n', ' ')], model='text-embedding-ada-002')
    embedding = raw['data'][0]['embedding']
    embeddings.append(embedding)

print('Embeddings successfully created!')

print('Updating MongoDB with the new embeddings')
count = 0
for doc in collection.find():
    doc['bioEmbedding'] = embeddings[count]
    collection.replace_one({'_id': doc['_id']}, doc)
    count += 1
