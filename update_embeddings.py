import os

import pymongo
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
client = pymongo.MongoClient(os.environ['MONGODB_URI'])
db = client.get_database('ai_ml_playground')
collection = db.get_collection('personnel')

personnel = collection.find()

print('***EMBEDDINGS UPDATE TOOL***')
print('Starting to update the embeddings for {} personnel'.format(collection.count_documents({})))
bios = []
for doc in personnel:
    bios.append(doc.get('bio').replace('\n', ' '))

embeddings = model.encode(bios, show_progress_bar=True)
print('Embeddings successfully created!')

print('Updating MongoDB with the new embeddings')
count = 0
for doc in collection.find():
    doc['bioEmbedding'] = embeddings[count].tolist()
    collection.replace_one({'_id': doc['_id']}, doc)
    count += 1
