import urllib.request, urllib.error, urllib.parse
from urllib.parse import quote
import json
import os
from pprint import pprint

REST_URL = "http://data.bioontology.org"
API_KEY = "b4bd0449-2ac7-48e1-9b64-afd369c0434d"

def get_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())


def run_class_search(search_terms, ontologies, exact_match):
    search_results = []
    for term in search_terms:
        encoded_term = quote(term)
        req_string_basic_ontologies = f"/search?q={encoded_term}&ontologies={ontologies}&require_exact_match={exact_match}"
        REQUEST_SUBSTR = req_string_basic_ontologies

        list_res = get_json(REST_URL + REQUEST_SUBSTR)["collection"]
        res_set = {
            "search_term" : term,
            "matches" : []
        }
        res_subset = []
        for query_res in list_res:
            res_subset.append({
                "@context": query_res.get("@context", None),
                "@id": query_res.get("@id", None),
                "definition": query_res.get("definition", None),
                "matchType": query_res.get("matchType", None),
                "obsolete": query_res.get("obsolete", None), 
                "ontologyType": query_res.get("ontologyType", None),
                "prefLabel": query_res.get("prefLabel", None),
                "synonym": query_res.get("synonym", None)
            })
        res_set['matches'] = res_subset
        search_results.append(res_set)
    return search_results

def get_num_unique_term_matches(multi_term, val_match):
    key_set = ['definition','prefLabel','synonym']
    terms = multi_term.split()
    num_terms = len(terms)
    num_hits = 0
    for term in terms:
        for key in key_set:
            res_key = val_match[key]
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

def main():
    path = os.path.join(os.path.dirname(__file__), 'classes_search_terms.txt')
    terms_file = open(path, "r")
    search_terms = []
    for line in terms_file:
        search_terms.append(line.strip())

    ontologies = "CHIRO,MONDO"
    exact_match = 'false'
    endpoint = 'search'
    output_file = f"bp_class_search_chiro_mondo_at_least_n_minus_one.jsonl"
    top_k = 20

    results_set =  run_class_search(search_terms, ontologies, exact_match)

    ## Select the entities that match a majority of multi-word terms: 2/3, 3/4, etc..
    with open(output_file, 'a') as file:
        for res in results_set:
            search_term = res['search_term']
            new_entry = {'search_term': search_term, 'endpoint': endpoint, 'ontologies': ontologies}
            new_entry_matches = []
            for m_val in res['matches']:
                num_terms, num_hits = get_num_unique_term_matches(search_term, m_val)
                if num_terms > 1 and num_hits >= num_terms - 1:
                    new_entry_matches.append(m_val)
                elif num_terms == 1 and num_hits > 0:
                    new_entry_matches.append(m_val)
            
            if len(new_entry_matches) > 0:
                new_entry['num_matches'] = len(new_entry_matches) if (len(new_entry_matches) <= top_k) else top_k
                new_entry['matches'] = new_entry_matches[:top_k]
                file.write(json.dumps(new_entry) + '\n')

if __name__ == "__main__":
    main()