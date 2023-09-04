import json
import os

import openai
import pymongo

openai.api_key = os.environ['OPENAI_API_KEY']
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
                'vector': openai.Embedding.create(input=[query], model='text-embedding-ada-002')['data'][0][
                    'embedding'],
                'k': 3,
                'path': 'bioEmbedding'
            }
        }
    },
    {
        '$project': {
            '_id': 0,
            'firstName': 1,
            'lastName': 1,
            'position': 1,
            'b/t': 1,
            'class': 1,
            'hometown': 1,
            'bio': 1,
            # 'score': {'$meta': 'searchScore'}
        }
    }
])

print('\n\n\n***RESULTS***')
chat = openai.ChatCompletion.create(
    model='gpt-4',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant answering questions about college baseball.'},
        {'role': 'user', 'content': 'Answer the following question:'
                                    + query
                                    + 'by using the following MongoDB document profile on the individual:\n\n'
                                    + json.dumps(results.next())},
    ]
)
print(chat.choices[0].message.content)
