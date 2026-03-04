db = db.getSiblingDB('ai-defra-search-agent');

db.knowledgeGroups.drop();
db.knowledgeGroups.insertOne(JSON.parse(fs.readFileSync('/docker-entrypoint-initdb.d/data/knowledgeGroups.json', 'utf8')));

db.knowledgeSnapshots.drop();
db.knowledgeSnapshots.insertOne(JSON.parse(fs.readFileSync('/docker-entrypoint-initdb.d/data/knowledgeSnapshots.json', 'utf8')));
