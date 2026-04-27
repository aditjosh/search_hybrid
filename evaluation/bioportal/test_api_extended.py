#!/usr/bin/env python3
"""
API test script — functional tests, precision, recall evaluation.

Golden set is loaded from golden_set.json.
Regenerate it with:
    python build_golden_set.py          # ~1000 entries (default)
    python build_golden_set.py --size 500

Usage:
    python test_api.py                              # functional tests only
    python test_api.py --pr                        # + precision, recall per endpoint
    python test_api.py --perf                       # + performance benchmark
    python test_api.py --all                        # everything
    python test_api.py --url http://host:8000 --all --verbose
    python test_api.py --wait                       # poll /health until indexing done
    python test_api.py --pr --max_queries 200      # limit number of queries
    python test_api.py --all --plot --plot-dir test_results/
    python test_api.py --all --plot --csv           # also save CSV files
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

try:
    import requests
except ImportError:
    print("ERROR: requests not installed — pip install requests")
    sys.exit(1)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"
WARN = "\033[93mWARN\033[0m"

passed = 0
failed = 0

_THIS_DIR = Path(__file__).parent

def section(title: str):
    print(f"\n── {title}")

def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [{PASS}] {label}" + (f": {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  [{FAIL}] {label}" + (f": {detail}" if detail else ""))
    return ok


def info(msg: str):
    print(f"  [{INFO}] {msg}")


def warn(msg: str):
    print(f"  [{WARN}] {msg}")


def post(url, path, payload, verbose=False):
    try:
        r = requests.post(f"{url}{path}", json=payload, timeout=60)
        if verbose:
            print(json.dumps(r.json(), indent=2))
        return r
    except Exception:
        return None


def get(url, path, verbose=False):
    try:
        r = requests.get(f"{url}{path}", timeout=30)
        if verbose:
            print(json.dumps(r.json(), indent=2))
        return r
    except Exception:
        return None


def fetch_config(url: str) -> dict:
    """Fetch /config for embedding into plot titles/captions."""
    r = get(url, "/config")
    if r and r.status_code == 200:
        return r.json()
    return {}


# ── Golden set ────────────────────────────────────────────────────────────────
def load_golden(path: Path = _THIS_DIR / "golden_set.jsonl") -> dict:
    """
    Load golden set from JSONL file. Returns dict with keys:
      - "single": list of {query, context, acceptable_labels, endpoint, note}
      - "meta":   stats dict
    """
    if not path.exists():
        warn(f"File not found at {path}")
        return {"single": [],  "meta": {}}

    with open(path) as f:
        data = [json.loads(line) for line in f]

    return {"single": data, "meta": {"total_single": len(data)}}

GOLDEN = load_golden(_THIS_DIR / "golden_set.jsonl")

def _matches(results: list, acceptable: List[str]) -> Optional[int]:
    """
    Input: 
        - results: List of items retrieved for endpoint query.
        - acceptable: List of items relevant to the endpoint query.
    Output:
        - num_hits: Number of retrieved items that match an item in the acceptable (relevant) list.
    """
    num_hits = 0
    exhausted = set()
    for i, r in enumerate(results):
        # Lowercase the returned label for case-insensitive substring matching
        eid = r.get("ontology_id", "").lower()
        # Accept if ANY of the acceptable ids was found
        for a in acceptable:
            if eid == a['@id'].lower() and not a['@id'].lower() in exhausted:
                num_hits += 1  # 1-based: first result in list is rank 1
                exhausted.add(a['@id'].lower())

    return num_hits # correct answer not found in the top-k results returned by the API

def _endpoint_stats(match_counts: List[Optional[int]], relevant_counts: List[Optional[int]],
                    precisions: List[Optional[int]], recalls: List[Optional[int]]) -> dict:
    """
    Inputs
        - match_counts: Number of relevant items retrieved for different endpoint queries. Includes null values.
        - relevant_counts: Number of relevant items for different endpoint queries.
    Outputs
        - num_queries: Number of endpoint queries, that is, the length of match_counts.
        - relevant_counts: Number of relevant items for different endpoint queries.
        - total_hits: Total number of relevant items retrieved in match_counts. Excludes null values.
        - total_misses: Total number of relevant items not retrieved in match_counts.
        - hits: Same as match_counts.
        - misses: Number of relevant items not included in the retrieved items for different endpoint queries.
        - precisions_at_k: Number of relevant items in top_k / topk retrieved items.
        - recalls_at_k: Number of relevant items in top_k / relevant items.
    """
    len_match_counts = len(match_counts)
    sum_accept_counts = sum(relevant_counts)
    if len_match_counts == 0:
        # Return the structure to print the report
        return {"num_queries": 0, "hits": 0, "misses": 0, "relevant_counts": 0,
            "total_hits": 0, "total_misses": 0, "precisions_at_k": 0, "recalls_at_k": 0}

    # Count queries where match count is defined (not None)
    total_hits = sum(r for r in match_counts if r)
    total_misses = sum_accept_counts - total_hits

    hit_counts = []
    miss_counts = []
    misses = 0
    for i in range(len_match_counts):
        m = match_counts[i]
        a = relevant_counts[i]
        if m:
            misses = a - m
        else:
            misses = a
        hit_counts.append(m)
        miss_counts.append(misses)

    return {"num_queries": len_match_counts, "hits": hit_counts, "misses": miss_counts, "relevant_counts": relevant_counts,
            "total_hits": total_hits, "total_misses": total_misses, "precisions_at_k": precisions, "recalls_at_k": recalls}

def test_health(url: str, verbose: bool) -> bool:
    section("Health check")
    r = get(url, "/health", verbose)
    if not check("GET /health reachable", r is not None and r.status_code == 200,
                 f"status={getattr(r,'status_code','err')}"):
        return False
    data = r.json()
    check("database_ready",  data.get("database_ready"),  str(data.get("database_ready")))
    check("retriever_ready", data.get("retriever_ready"), str(data.get("retriever_ready")))
    check("reranker_ready",  data.get("reranker_ready"),  str(data.get("reranker_ready")))
    indexing = data.get("indexing_complete", False)
    if not indexing:
        print(f"  [{SKIP}] indexing_complete=False — search endpoints return 503 until ready")
    else:
        check("indexing_complete", True, "True")
    return indexing

# ── Evaluation ───────────────────────────────────────────────────────
def _eval_single(url: str, entry: dict, top_k: int):
    """Evaluate one single entry via /map/concept or /map/search. Returns (rank_or_None, top_id)."""
    endpoint = "/map/search" if entry.get("endpoint") != "search" else "/map/search"
    payload = {"text": entry["search_term"], "max_results": top_k}
    if entry.get("context"):
        payload["context"] = entry["context"]
    if entry.get("ontologies"):
        payload["ontologies"] = entry["ontologies"]

    r = post(url, endpoint, payload)
    if r is None or r.status_code != 200:
        return None, "—"
    results = r.json().get("results", [])

    relevant_items = entry["matches"]
    n_matches = _matches(results, relevant_items)
    recall_at_k = n_matches/len(relevant_items) if relevant_items else 0.0
    precision_at_k = n_matches/len(results) if results else 0.0
    top_id = results[0]["ontology_id"] if results else None
    return n_matches, top_id, precision_at_k, recall_at_k

def test_precision_recall(url: str, top_k: int = 5, max_queries: Optional[int] = None, 
                  query_stats_file: str = None, verbose: bool = False):
    golden_single = GOLDEN.get("single", [])

    if not golden_single:
        warn("No golden set — skipping tests.")
        return None

    total_single = len(golden_single)
    section(
        f"Performance evaluation  "
        f"({total_single} single "
    )

    meta = GOLDEN.get("meta", {})
    if meta.get("breakdown"):
        info("Golden set breakdown: " + "  ".join(f"{k}={v}" for k, v in meta["breakdown"].items()))

    # Per-endpoint rank accumulation.
    # ep_ranks collects the raw rank (int or None) for every evaluated query,
    # keyed by which endpoint was used.  After the loop, _endpoint_stats() converts
    # these lists into Hit@k counts and MRR — one stats dict per endpoint + overall.
    ep_matches: Dict[str, List[Optional[int]]] = {"concept": [], "search": []}
    ep_accepted_counts: Dict[str, List[Optional[int]]] = {"concept": [], "search": []}
    ep_precisions: Dict[str, List[Optional[int]]] = {"concept": [], "search": []}
    ep_recalls: Dict[str, List[Optional[int]]] = {"concept": [], "search": []}
    per_query_rows: List[dict] = []  # for CSV export — one row per query
    errors = 0

    # ── Single entries (/map/concept and /map/search) ────────────────────────
    info(f"Evaluating {total_single} single queries...")
    for _qi, entry in enumerate(golden_single):
        n_matches, top_id, precision_at_k, recall_at_k = _eval_single(url, entry, top_k)
        if not verbose and (_qi + 1) % 10 == 0:
            print(f"  ... {_qi + 1}/{total_single} queries evaluated", flush=True)
        ep = entry.get("endpoint", "concept")  # "concept" or "search"

        if n_matches is None and top_id is None:
            warn(f"'{entry['search_term']}' → request failed")
            errors += 1
            ep_matches[ep].append(None)
        else:
            ep_matches[ep].append(n_matches)
        ep_accepted_counts[ep].append(entry['num_matches'])
        ep_precisions[ep].append(precision_at_k)
        ep_recalls[ep].append(recall_at_k)

        # Build the per-query CSV row.
        per_query_rows.append({
            "search_term":       entry["search_term"],
            "endpoint":    f"/map/{ep}",
            "num_of_matches":    n_matches if n_matches else "",         # empty string = not found
            "not_found":   "1" if not n_matches else "0",
            "num_of_matches":    "1" if n_matches else "0",       
            "top_id":   top_id,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k
        })

        if verbose:
            mark = PASS if n_matches == 1 else (WARN if n_matches else FAIL)
            n_matches_str = f"n_matches={n_matches}" if n_matches else "NOT FOUND"
            ctx_hint = f" [ctx]" if entry.get("context") else ""
            print(f"  [{mark}] [{ep}]{ctx_hint} '{entry['search_term']}' → {n_matches_str}  top='{top_id}'")

    # "overall" merges all three endpoint rank lists into one and runs the same computation,
    # giving a single aggregate score across all N evaluated queries.
    accept_count = []
    for ep in ("concept", "search"):
        accept_count = []
        ep_stats = {ep: _endpoint_stats(ep_matches[ep], ep_accepted_counts[ep], ep_precisions[ep], ep_recalls[ep]) for ep in ("concept", "search")}
    
    all_ep_matches = ep_matches["concept"] + ep_matches["search"]
    all_ep_accepted_counts = ep_accepted_counts["concept"] + ep_accepted_counts["search"]
    all_ep_precisions = ep_precisions["concept"] + ep_precisions["search"]
    all_ep_recalls = ep_recalls["concept"] + ep_recalls["search"]
    overall = _endpoint_stats(all_ep_matches, all_ep_accepted_counts, all_ep_precisions, all_ep_recalls)


    def _pct(stats, key, n_fallback=None):
        """
        Example input:
            stats = {'num_queries': 0, 'num_expected_matches': 0, 'hits': 0, 'misses': 0}
            stats = {'num_queries': 4, 'num_expected_matches': [3, 4, 5, 2], 'total_hits': 7, 'hits': [None, 3, 4, None], 'misses': [3, 4, 5, 7]}
        Example output:
            [(hit_1,miss_1), (hit_2,miss_2)...(hit_n,miss_n)]
        """
        stats_key = stats[key]
        if stats_key:
            relevant_counts = stats_key["relevant_counts"] or n_fallback or 1
            if isinstance(relevant_counts, int):
                return []
            elif isinstance(relevant_counts, list):
                hits = stats_key['hits']
                misses = stats_key['misses']
                hits_pct = []
                misses_pct = []
                for c,exp_m in enumerate(relevant_counts):
                    curr_hit = hits[c] if hits[c] else 0
                    curr_hit_pc = (curr_hit/exp_m)* 100 
                    hits_pct.append(curr_hit_pc)
                    curr_miss_pc = ((misses[c])/exp_m)* 100
                    misses_pct.append(curr_miss_pc)

                return list(zip(hits_pct, misses_pct))
        else:
            return f"{'—':>28}"

    # ── Summary table ─────────────────────────────────────────────────────────
    def print_table(file_path, cols, key, ep_stats):
        with open(file_path, "a") as fp:
            s_key = _pct(ep_stats, key)
            hits_rows = []
            misses_rows = []
            hits = []
            misses = []
            for xi, xs in enumerate(s_key):
                row_pf = f"{xi}      "
                row = f"    {str('hits').capitalize():<12}"
                if isinstance(xs,tuple):
                    row += f"{' ':>24}" + str(xs[0])
                    hits_rows.append(row_pf + row)
                    row = f"    {str('misses').capitalize():<12}"
                    row += f"{' ':>24}" + str(xs[1])
                    misses_rows.append(row_pf + row)
                    hits.append(xs[0])
                    misses.append(xs[1])

            if len(hits_rows) > 0:
                header = f"{'Query_Idx'}  {'Metric':<16}" + "".join(f"{lbl:>30}" for _, lbl in cols)
                print(f"\n{header}", file=fp)
                print(f"{'-'*(32 + 28*len(cols))}", file=fp)
                for i,v in enumerate(hits_rows):
                    print(hits_rows[i], file=fp)
                    print(misses_rows[i], file=fp)
                print(f"{'-'*(32 + 28*len(cols))}", file=fp)
        return hits, misses

    
    if query_stats_file:
        ## Search
        cols = [("search",  f"/map/search (n={ep_stats['search']['num_queries']})")]
        hits_search, misses_search = print_table(query_stats_file, cols, 'search', ep_stats)

        ## Concept
        cols = [("concept",  f"/map/concept (n={ep_stats['concept']['num_queries']})")]
        hits_concept, misses_concept = print_table(query_stats_file, cols, 'concept', ep_stats)

    ## Average precision and recall
    prec_q_05_at_k = round(np.quantile(all_ep_precisions, 0.25), 3)
    prec_q_50_at_k = round(np.quantile(all_ep_precisions, 0.5), 3)
    prec_q_95_at_k = round(np.quantile(all_ep_precisions, 0.95), 3)
    recall_q_05_at_k = round(np.quantile(all_ep_recalls, 0.25), 3)
    recall_q_50_at_k = round(np.quantile(all_ep_recalls, 0.5), 3)
    recall_q_95_at_k = round(np.quantile(all_ep_recalls, 0.95), 3)

    avg_precision_at_k = round(sum(all_ep_precisions)/len(all_ep_precisions), 3)
    avg_recall_at_k = round(sum(all_ep_recalls)/len(all_ep_recalls), 3)
    print(f"    {'Average precision@k':<12}{avg_precision_at_k:>28}")
    print(f"    {'5% quantile precision@k':<5}{prec_q_05_at_k:>22}")
    print(f"    {'50% quantile precision@k':<6}{prec_q_50_at_k:>22}")
    print(f"    {'95% quantile precision@k':<6}{prec_q_95_at_k:>23}")
    print(f"    {'Average recall@k':<12}{avg_recall_at_k:>31}")
    print(f"    {'5% quantile recall@k':<5}{recall_q_05_at_k:>25}")
    print(f"    {'50% quantile recall@k':<6}{recall_q_50_at_k:>26}")
    print(f"    {'95% quantile recall@k':<6}{recall_q_95_at_k:>26}")

    if errors:
        print(f"  {'Errors':<12}{errors:>28}")

    n_total = sum(overall["relevant_counts"])
    #print(overall)

    #check(f"Hit@1 ≥ 50% (overall, n={n_total})", overall["hit_at"][1] / n_total >= 0.50 if n_total else False,
    #      f"{overall['hit_at'][1]/n_total*100:.1f}%" if n_total else "no data")
    #check(f"Hit@3 ≥ 70% (overall)", overall["hit_at"][3] / n_total >= 0.70 if n_total else False,
    #      f"{overall['hit_at'][3]/n_total*100:.1f}%" if n_total else "no data")
    check(f"Hits ≥ 80% (overall)", overall["total_hits"] / n_total >= 0.80 if n_total else False,
          f"{overall['total_hits']/n_total*100:.2f}%" if n_total else "no data")
    check(f"Misses <= 20% (overall)", overall["total_misses"] / n_total <= 0.20 if n_total else False,
          f"{overall["total_misses"] /n_total*100:.2f}%" if n_total else "no data")

    return {
        "ep_stats":     ep_stats,
        "ep_matches":   ep_matches,
        "ep_accepted_counts":   ep_accepted_counts,
        "overall":      overall,
        "per_query":    per_query_rows,
        "errors":       errors,
        "top_k":        top_k,
        "hit_at":       overall["hits"],
        "n":            n_total
    }

    # ── Config snapshot ───────────────────────────────────────────────────────
    if server_config:
        path = os.path.join(out_dir, f"config_{ts}.json")
        with open(path, "w") as f:
            json.dump(server_config, f, indent=2)
        saved.append(path)

# ── Plotting ──────────────────────────────────────────────────────────────────
def _config_subtitle(cfg: dict) -> str:
    """One-line config string for figure subtitles."""
    if not cfg:
        return ""
    r = cfg.get("retrieval", {})
    rk = cfg.get("reranking", {})
    parts = []
    if rk.get("reranker_type"):
        parts.append(f"reranker={rk['reranker_type']}")
    if r.get("embedding_model"):
        m = r["embedding_model"].split("/")[-1]
        parts.append(f"model={m}")
    if r.get("vector_backend"):
        parts.append(f"backend={r['vector_backend']}")
    return "  |  ".join(parts)


def save_plots(pr_data: Optional[dict], out_dir: str, fmt: str,
               server_config: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        warn("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    cfg_sub = _config_subtitle(server_config)

    # ── Figure ───────────────────────────────────────────────────────
    if pr_data:
        ep_stats = pr_data["ep_stats"]
        overall  = pr_data["overall"]
        n_total  = overall["total_hits"]
        hits_search = pr_data['hits_search']
        misses_search = pr_data['misses_search']
        data_list = [hits_search, misses_search]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(
            f"Ontology Mapping API — (N={len(data_list[0]) + len(data_list[1])} queries evaluated)",
            fontsize=13, fontweight="bold"
        )
        sns.violinplot(data=data_list, ax=ax, palette="Pastel1", fill=False)
        #axes[0].set_title("Boxplot of 2 Arrays")
        ax.set_xticklabels(["Hits", "Misses"])
        fig.subplots_adjust(top=0.94)

        plt.tight_layout()
        path = os.path.join(out_dir)
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

    if saved:
        print(f"\n  Plots saved to: {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the Ontology Mapping API")
    parser.add_argument("--url",           default="http://localhost:8000")
    parser.add_argument("--verbose",       action="store_true", help="Print per-query results")
    parser.add_argument("--wait",          action="store_true", help="Poll /health until indexing is complete")
    parser.add_argument("--pr",            action="store_true", help="Run precision,recall evaluation per endpoint")
    parser.add_argument("--batch-sizes",   action="store_true", help="Run batch size variation test")
    parser.add_argument("--all",           action="store_true", help="Run all tests")
    parser.add_argument("--max_queries",   type=int, default=None, help="Limit analysis to N single queries (smoke-test)")
    parser.add_argument("--n-requests",    type=int, default=100,  help="Requests for perf benchmark (default 100)")
    parser.add_argument("--concurrency",   type=int, default=4,    help="Concurrent threads for perf benchmark (default 4)")
    parser.add_argument("--plot",          action="store_true",    help="Save plots")
    parser.add_argument("--stats_file",    default="test_results/query_stats.txt",   help="Save results in the text file")
    parser.add_argument("--plot-format",   default="png", choices=["png", "pdf"])
    parser.add_argument("--plot-dir",      default="test_results", help="Output directory for plots/CSV (default: test_results/)")
    parser.add_argument("--data-file",     type=str, default=None, help="File containing prepared labeled dataset.")
    
    args = parser.parse_args()

    if args.all:
        args.pr = args.perf = args.batch_sizes = True

    url = args.url.rstrip("/")
    print(f"Testing API: {url}")

    # Fetch server config early for plot/CSV metadata
    server_config = fetch_config(url)
    if server_config:
        rc = server_config.get("reranking", {})
        rv = server_config.get("retrieval", {})
        info(f"Server config: reranker={rc.get('reranker_type','?')}  "
             f"model={rv.get('embedding_model','?')}  backend={rv.get('vector_backend','?')}")

    indexing_ready = test_health(url, args.verbose)

    if args.wait and not indexing_ready:
        print(f"\n  Waiting for indexing (polling every 30s)...")
        while not indexing_ready:
            time.sleep(30)
            r = get(url, "/health")
            if r and r.status_code == 200:
                indexing_ready = r.json().get("indexing_complete", False)
                print(f"  [{INFO}] indexing_complete={indexing_ready}")
        print(f"  [{PASS}] Indexing complete")

    global GOLDEN
    if args.data_file:
        golden_data_filename = str(args.data_file)
    else:
        golden_data_filename = 'bp_class_search_chiro_mondo_at_least_n_minus_one.jsonl'
    GOLDEN = load_golden(_THIS_DIR / golden_data_filename)

    if indexing_ready:
        pr_data = None

        if args.pr:
            max_results = 20
            if args.stats_file:
                ## Remove previous file if found
                if os.path.exists(args.stats_file):
                    os.remove(args.stats_file)
                pr_data = test_precision_recall(url, verbose=args.verbose, query_stats_file=args.stats_file, 
                                        top_k=max_results, max_queries=args.max_queries)
            else:
                pr_data = test_precision_recall(url, verbose=args.verbose, top_k=max_results, max_queries=args.max_queries)
            
        if args.plot and (pr_data):
            section("Saving plots")
            save_plots(pr_data, args.plot_dir, args.plot_format, server_config)

    else:
        print(f"\n  [{SKIP}] Search tests skipped — indexing not complete (use --wait)")

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"RESULT: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  ALL PASSED")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
