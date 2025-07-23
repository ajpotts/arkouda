#!/usr/bin/env python3
import os
import re
import json

GRAPH_INFRA_DIR = "benchmark_v2/graph_infra"
OUTPUT_JSON = "benchmark_v2/datdir/configs/field_lookup_map.json"

# Group inference (extend as needed)
GROUP_MAP = {
    "groupby": "GroupBy_Creation",
    "str-groupby": "GroupBy_Creation",
    "bigint-groupby": "GroupBy_Creation",
    "coargsort": "Arkouda_CoArgSort",
    "str-coargsort": "Arkouda_CoArgSort",
    "bigint-coargsort": "Arkouda_CoArgSort",
    "sort-cases": "AK_Sort_Cases",
    "scan": "scan",
    "reduce": "reduce",
    "aggregate": "aggregate",
    "stream": "stream",
    "argsort": "argsort",
    "gather": "gather",
    "scatter": "scatter",
    "str-gather": "gather",
    # add others as needed
}

def infer_regex(benchmark_name: str, field: str) -> str:
    base_bench = benchmark_name.replace("str-", "").replace("bigint-", "")

    # str-gather simple default case
    if benchmark_name == "str-gather":
        return f"bench_{base_bench}\\[str\\]"

    # Handle array cases
    if "array" in field:
        m = re.search(r"(\d+)-array", field)
        if m:
            num = m.group(1)
            if benchmark_name.startswith("str-"):
                dtype = "str"
            elif benchmark_name.startswith("bigint-"):
                dtype = "bigint"
            else:
                dtype = "(?:int64|float64|bool|uint64)"
            return f"bench_{base_bench}\\[{dtype}-{num}\\]"

    # Fallback for gather/str-gather without "array" naming
    if benchmark_name in {"gather", "str-gather"}:
        if benchmark_name.startswith("str-"):
            return f"bench_{base_bench}\\[str\\]"
        else:
            return f"bench_{base_bench}\\[(?:int64|float64|bool|uint64)\\]"

    # Handle sort-cases (heuristic)
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

    # Handle reduce/scan/aggregate
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
                "benchmark_name": benchmark_name.replace("str-", "").replace("bigint-", "").replace("-", "_"),
                "lookup_path": lookup_path,
                "lookup_regex": regex,
            }

    return field_lookup_map

DEFAULT_BENCHMARKS = [
    "stream", "argsort", "gather", "scatter", "dataframe", "bigint_stream",
    "str-gather"  # ✅ now included
]

def add_default_mappings(field_lookup_map):
    for b in DEFAULT_BENCHMARKS:
        if b not in field_lookup_map:
            base_bench = b.replace("str-", "").replace("bigint-", "").replace("-", "_")
            field_lookup_map[b] = {
                "Average rate =": {
                    "group": "",
                    "name": f"bench_{b}",
                    "benchmark_name": b.replace("-", "_"),
                    "lookup_path": ["extra_info", "transfer_rate"],
                    "lookup_regex": f"bench_{base_bench}\\[[\\w\\d]*\\]",

                },
                "Average time =": {
                    "group": "",
                    "name": f"bench_{b}",
                    "benchmark_name": b.replace("-", "_"),
                    "lookup_path": ["stats", "mean"],
                    "lookup_regex": f"bench_{base_bench}\\[[\\w\\d]*\\]",
                },
            }
    return field_lookup_map


AGGREGATE_OPS = [
    "prod", "sum", "mean", "min", "max", "argmin", "argmax",
    "any", "all", "xor", "and", "or", "nunique"
]

def add_aggregate_ops(field_lookup_map):
    if "aggregate" not in field_lookup_map:
        field_lookup_map["aggregate"] = {}

    for op in AGGREGATE_OPS:
        for t in ["time", "rate"]:
            lookup_path = (
                ["extra_info", "transfer_rate"] if t == "rate" else ["stats", "mean"]
            )
            field_lookup_map["aggregate"][f"Aggregate {op} Average {t} ="] = {
                "group": "GroupBy.aggregate",
                "name": f"bench_aggregate[{op}]",
                "benchmark_name": "aggregate",
                "lookup_path": lookup_path,
                "lookup_regex": f"bench_aggregate\\[{op}\\]",
            }
    return field_lookup_map

import logging
def add_str_bigint_cases(field_lookup_map):
    for case in ["str-groupby", "str-coargsort", "bigint-groupby", "bigint-coargsort"]:
        if case not in field_lookup_map:
            logging.warning(f"{case} perfkeys exist but no mapping was built")
    return field_lookup_map

def main():
    field_lookup_map = build_field_lookup_map()
    field_lookup_map = add_default_mappings(field_lookup_map)  # previous fix
    field_lookup_map = add_aggregate_ops(field_lookup_map)     # ✅ add this
    field_lookup_map = add_str_bigint_cases(field_lookup_map)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(field_lookup_map, f, indent=2)
    print(f"✅ Updated {OUTPUT_JSON} with {len(field_lookup_map)} benchmarks.")


if __name__ == "__main__":
    main()
