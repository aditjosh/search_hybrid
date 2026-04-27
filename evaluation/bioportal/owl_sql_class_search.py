import sqlite3, json, os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Common namespaces in OWL/RDF
NAMESPACES = {
    'rdf':  'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'owl':  'http://www.w3.org/2002/07/owl#',
    'xsd':  'http://www.w3.org/2001/XMLSchema#',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
    'obo': 'http://purl.obolibrary.org/obo/'
}

# Extract entitites from OWL and store in SQL db
class OwlDB():
    def __init__(self):
        for prefix, uri in NAMESPACES.items():
            ET.register_namespace(prefix, uri)
        
    def extract_entities_from_owl(self, owl_file: str) -> List[Dict]:
        """Extract classes with label and definition using pure stdlib XML parsing."""
        tree = ET.parse(owl_file)
        root = tree.getroot()

        entities = []

        # Find all OWL classes
        for cls in root.findall('.//owl:Class', NAMESPACES):
            entity_id = cls.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about') or \
                        cls.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID')

            if not entity_id:
                continue

            # Get label
            label_elem = cls.find('rdfs:label', NAMESPACES)
            label = label_elem.text.strip() if label_elem is not None and label_elem.text else None

            # Get definition (try rdfs:comment, then skos:definition)
            #def_elem = cls.find('rdfs:comment', NAMESPACES) or cls.find('obo:IAO_0000115', NAMESPACES)
            def_elem = cls.find('obo:IAO_0000115', NAMESPACES)
            definition = def_elem.text.strip() if def_elem is not None and def_elem.text else None

            if label or definition:   # Only keep entities with some info
                entities.append({
                    'entity_id': entity_id,
                    'label': label,
                    'definition': definition
                })

        return entities

    def create_database(self, db_file: str):
        """Create SQLite table."""
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT UNIQUE,
                label TEXT,
                definition TEXT
            )
        ''')
        conn.commit()
        conn.close()


    def insert_entities(self, db_file: str, entities: List[Dict]):
        """Insert extracted entities into SQLite."""
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        for ent in entities:
            cursor.execute('''
                INSERT OR REPLACE INTO entities (entity_id, label, definition)
                VALUES (?, ?, ?)
            ''', (ent['entity_id'], ent['label'], ent['definition']))
        
        conn.commit()
        conn.close()
        print(f"Inserted {len(entities)} entities into SQLite database.")


# Search the SQL db containing OWL entities
class OwlDBSearch():
    def __init__(self, db_file):
        self.db_file = db_file

    def search_definitions(self, query: str, exact_match: bool = False, case_sensitive: bool = False) -> List[Dict]:
        """Search using pure Python loop (no SQL LIKE or full-text)."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT entity_id, label, definition FROM entities")
        rows = cursor.fetchall()
        conn.close()

        results = []
        frozen_results_set = set()
        search_term = query if case_sensitive else query.lower()

        for row in rows:
            entity_id, label, definition = row
            d = { 'entity_id': entity_id, 'label': label, 'definition': definition}
            text_to_search = (definition or "") + " " + (label or "")
            
            if not case_sensitive:
                text_to_search = text_to_search.lower()

            # Exact match
            if search_term in text_to_search:
                if not frozenset(d.items()) in frozen_results_set:
                    results.append(d)

            # Include additional matches if exact match not True
            # (1) Strategy: Match without the last term
            if not exact_match:
                if len(search_term) > 1 and (search_term[:-1] in text_to_search):
                    if not frozenset(d.items()) in frozen_results_set:
                        results.append(d)

        return results


def extract_and_create_database(owl_file, db_file):
    owl_db = OwlDB()
    print("Parsing OWL ontology...")
    entities = owl_db.extract_entities_from_owl(owl_file)
    owl_db.create_database(db_file)
    owl_db.insert_entities(db_file, entities)
    print(f"Created {db_file}")

def search_db(db_file, search_query = "hereditary spastic paraplegia", exact_match = False):
    if search_query and search_query != '':
        search_db = OwlDBSearch(db_file)
        matches = search_db.search_definitions(query=search_query, exact_match=exact_match, case_sensitive=False)
        
        print(f"\n Found {len(matches)} matching entities.")
        results = []
        for m in matches:  # limit output
            results.append({
                '@id': m['entity_id'],
                'label': m['label'], 
                'definition': m['definition'] if m['definition'] else ''
                })
        return results

def get_num_unique_term_matches(multi_term, result_dict):
    key_set = ['definition', 'label']
    terms = multi_term.split()
    num_terms = len(terms)
    num_hits = 0
    for term in terms:
        for key in key_set:
            res_key = result_dict[key]
            if res_key is not None:
                if isinstance(res_key, list):
                    if term in " ".join(res_key):
                        num_hits += 1
                        break
                if isinstance(res_key, str):
                    if term in res_key:
                        num_hits += 1
                        break
    return num_terms, num_hits

def remove_unwanted_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = [w for w in tokens if w.lower() not in stop_words]
    return " ".join(filtered_text)

def search_results_to_jsonl(output_file, search_results, search_term, endpoint, ontologies, exact_match=False, top_k=20):
    with open(output_file, 'a') as file:
        #print(search_results)
        new_entry_matches = []
        for res in search_results:
            new_entry = {'search_term': search_term, 'endpoint': endpoint, 'ontologies': ontologies}
            
            num_terms, num_hits = get_num_unique_term_matches(search_term, res)
            
            if num_terms > 1 and num_hits >= num_terms - 1:
                new_entry_matches.append(res)
            elif num_terms == 1 and num_hits > 0:
                new_entry_matches.append(res)
        
        if len(new_entry_matches) > 0:
            new_entry['num_matches'] = len(new_entry_matches) if (len(new_entry_matches) <= top_k) else top_k
            new_entry['matches'] = new_entry_matches[:top_k]
            file.write(json.dumps(new_entry) + '\n')

def setup_nltk():
    os.environ['NLTK_DATA'] = '/nfsdata02/ajoshi2/pyenv/nltk_data'
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# ====================== MAIN ======================
if __name__ == "__main__":
    #owl_file = "utilities/owl_app/ontologies/mondo-base.owl"
    setup_nltk()
    path = os.path.join(os.path.dirname(__file__), 'classes_search_terms.txt')
    terms_file = open(path, "r")
    search_terms = []
    for line in terms_file:
        clean_line = remove_unwanted_words(line.strip())
        search_terms.append(clean_line)

    db_files = [
        ("utilities/owl_app/ontologies/mondo-base.db","MONDO"),
        ("utilities/owl_app/ontologies/cido.db","CIDO")
    ]
    exact_match_val = True
    file_search_results_jsonl = f"owldb_class_search_cido_mondo_exact_match_{str(exact_match_val).lower()}_at_least_n_minus_one.jsonl"
    endpoint = "search"
    # extract_and_create_database(owl_file, db_file)

    for search_term in search_terms:
        print(f"Search term: {search_term}")
        search_results = []
        ontologies = "CIDO,MONDO"
        for db_file in db_files:
            #print(f"File: {db_file[0]}")
            search_results.extend(search_db(db_file[0], search_query = search_term, exact_match = exact_match_val))
        
        search_results_to_jsonl(file_search_results_jsonl, search_results, search_term, endpoint, ontologies, top_k=500)
    #print(search_results)


    