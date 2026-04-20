from flask import Flask, render_template, request
from owlready2 import *
import os, json
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

app = Flask(__name__)

class OWLSearch:
    def __init__(self, db_file):
        self.db_file = db_file
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT entity_id, label, definition FROM entities")
        rows = cursor.fetchall()
        conn.close()
        self.db_rows = rows

    def search(self, query: str, case_sensitive: bool = False):
        """Search using pure Python loop (no SQL LIKE or full-text)."""
        results = []
        search_term = query if case_sensitive else query.lower()

        for row in self.db_rows:
            entity_id, label, definition = row
            text_to_search = (definition or "") + " " + (label or "")
            
            if not case_sensitive:
                text_to_search = text_to_search.lower()

            if search_term in text_to_search:
                results.append({
                    'entity_id': entity_id,
                    'label': label,
                    'definition': definition
                })

        return results


ontology_db_file = 'ontologies/mondo-base.sqldb'
owl_search = OWLSearch(ontology_db_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    query_res = []
    resp_list = []
    search_term = ""
    if request.method == 'POST':
        search_term = request.form['query']
        if search_term:
            query_res = owl_search.search(query=search_term)
    
            if len(query_res) > 0:
                resp_list = json.dumps(query_res, indent=4)

    return render_template('index.html', items=resp_list, query=search_term)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
