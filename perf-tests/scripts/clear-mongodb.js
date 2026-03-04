db = db.getSiblingDB('ai-defra-search-agent')

print('Clearing MongoDB collections...')

db.knowledgeGroups.deleteMany({})
db.knowledgeSnapshots.deleteMany({})

print('MongoDB collections cleared successfully')
