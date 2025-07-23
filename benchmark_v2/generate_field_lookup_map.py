#!/usr/bin/env python3
import os
import re
import json

GRAPH_INFRA_DIR = "benchmark_v2/graph_infra"
OUTPUT_JSON = "benchmark_v2/graph_infra/field_lookup_map.json"

# Group inference (extend as needed)
GROUP_MAP = {
    "groupby": "GroupBy_Creation",
    "str-groupby": "GroupBy_Creation",
    "coargsort": "Arkouda_CoArgSort",
    "str-coargsort": "Arkouda_CoArgSort",
    "sort-cases": "AK_Sort_Cases",
    "scan": "scan",
    "reduce": "reduce",
    "aggregate": "aggregate",
    "stream": "stream",
    "argsort": "argsort",
    "gather": "gather",
    "scatter": "scatter",
    # add others as needed
}

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Union

benchmark_dir = os.path.dirname(__file__)
util_dir = os.path.join(benchmark_dir, "..", "server_util", "test")
sys.path.insert(0, os.path.abspath(util_dir))

logging.basicConfig(level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dat-dir",
        default=os.path.join(benchmark_dir, "datdir"),
        help="Directory with .dat files stored",
    )
    parser.add_argument("--graph-dir", help="Directory to place generated graphs")
    parser.add_argument(
        "--graph-infra",
        default=os.path.join(benchmark_dir, "graph_infra"),
        help="Directory containing graph infrastructure",
    )
    parser.add_argument("--platform-name", default="", help="Test platform name")
    parser.add_argument("--description", default="", help="Description of this configuration")
    parser.add_argument("--annotations", default="", help="File containing annotations")
    parser.add_argument("--configs", help="comma seperate list of configurations")
    parser.add_argument("--start-date", help="graph start date")
    parser.add_argument("--benchmark-data", help="the benchnmark output data in json format.")

    return parser

def infer_regex(benchmark_name: str, field: str) -> str:
    """Infer a regex for JSON benchmark names based on perfkey field names."""
    base_bench = benchmark_name.replace("str-", "")

    # Handle array cases (groupby, coargsort, etc.)
    if "array" in field:
        m = re.search(r"(\d+)-array", field)
        if m:
            num = m.group(1)
            dtype = "str" if benchmark_name.startswith("str-") else "(?:int64|float64|bool|uint64|bigint|mixed)"
            return f"bench_{base_bench}\\[{dtype}-{num}\\]"

    # Handle sort-cases (very heuristic; adjust as needed)
    if benchmark_name == "sort-cases":
        if "RMAT" in field:
            return r"bench_rmat\[[\w\d]+\]"
        if "block-sorted" in field:
            return r"bench_block_sorted\[[\w\d]+\]"
        if "refinement" in field:
            return r"bench_refinement\[[\w\d]+\]"
        if "datetime" in field:
            return r"bench_time_like\[[\w\d]+\]"
        if "IP" in field:
            return r"bench_ip_like\[[\w\d]+\]"
        if "power" in field or "uniform" in field:
            return r"bench_sort-cases\[[\w\d]*\]"

    # Handle reduce/scan/aggregate (generic fallback)
    if benchmark_name in {"reduce", "scan", "aggregate"}:
        op = field.split()[0]
        return f"bench_{benchmark_name}\\[{op}\\]"

    # Fallback
    return f"bench_{base_bench}\\[[\\w\\d]*\\]"


def get_header_fields_from_directory(directory_path):
    """Load perfkeys headers into a dict."""
    file_contents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".perfkeys"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                key = re.search(r"([\w\-_]+)\.perfkeys", filename)[1]
                file_contents[key] = lines
    return file_contents


def build_field_lookup_map():
    headers = get_header_fields_from_directory(GRAPH_INFRA_DIR)
    field_lookup_map = {}

    for benchmark_name, fields in headers.items():
        field_lookup_map[benchmark_name] = {}
        for field in fields:
            if field == "# Date":
                continue
            regex = infer_regex(benchmark_name, field)
            lookup_path = ["extra_info", "transfer_rate"] if "rate" in field else ["stats", "mean"]

            field_lookup_map[benchmark_name][field] = {
                "group": GROUP_MAP.get(benchmark_name, ""),
                "name": "",
                "benchmark_name": benchmark_name.replace("str-", "").replace("-", "_"),
                "lookup_path": lookup_path,
                "lookup_regex": regex,
            }

    return field_lookup_map


def main():

    parser = create_parser()
    args, client_args = parser.parse_known_args()
    args.graph_dir = args.graph_dir or os.path.join(args.dat_dir, "html")
    configs_dir = os.path.join(args.dat_dir, "configs")

    os.makedirs(configs_dir, exist_ok=True)

    lookup_map_path = configs_dir + "/field_lookup_map.json"

    field_lookup_map = build_field_lookup_map()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(field_lookup_map, f, indent=2)
    print(f"Updated {OUTPUT_JSON} with {len(field_lookup_map)} benchmarks.")


if __name__ == "__main__":
    main()

