import os

import pymongo
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
client = pymongo.MongoClient(os.environ['MONGODB_URI'])
db = client.get_database('ai_ml_playground')
collection = db.get_collection('personnel')

print('***Personnel Vector Search Tool***')
query = input('Query: ')
results = collection.aggregate([
    {
        '$search': {
            'index': 'default',
            'knnBeta': {
                'vector': model.encode([query])[0].tolist(),
                'k': 3,
                'path': 'bioEmbedding'
            }
        }
    },
    {
        '$project': {
            '_id': 1,
            'firstName': 1,
            'lastName': 1,
            'score': {'$meta': 'searchScore'}
        }
    }
])

print('\n\n\n***RESULTS***')
for doc in results:
    print(doc, '\n')
