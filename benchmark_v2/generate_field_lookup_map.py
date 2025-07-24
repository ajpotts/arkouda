import json
import logging

#!/usr/bin/env python3
import os
import re

# Aggregate operations explicitly defined
import arkouda as ak

GRAPH_INFRA_DIR = "benchmark_v2/graph_infra"
OUTPUT_JSON = "benchmark_v2/datdir/configs/field_lookup_map.json"

# Benchmarks that just need default Average rate/time keys
DEFAULT_BENCHMARKS = [
    "stream",
    "argsort",
    "str-argsort",
    "gather",
    "str-gather",
    "scatter",
    "dataframe",
    "bigint_stream",
    "flatten",
    "noop",
    "split",
]

# Group inference (update as needed)
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
    "aggregate": "GroupBy.aggregate",
    "stream": "stream",
    "argsort": "argsort",
    "str-argsort": "argsort",
    "gather": "gather",
    "str-gather": "gather",
    "scatter": "scatter",
    "dataframe": "dataframe",
    "bigint_stream": "stream",
    "flatten": "flatten",
}

AGGREGATE_OPS = ak.GroupBy.Reductions


def infer_regex(benchmark_name: str, field: str) -> str:
    """Infer a regex for JSON benchmark names based on perfkey field names."""
    base_bench = re.sub(r"^(str|bigint)(?:_|-)", "", benchmark_name)

    if "array_transfer" in base_bench:
        if "to_ndarray" in field:
            base_bench = base_bench + "_tondarray"
        elif "ak.array" in field:
            base_bench = base_bench + "_akarray"

    # Groupby & Coargsort (with array counts)
    if "array" in field and any(k in benchmark_name for k in ["groupby", "coargsort"]):
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

    # CSV Read/Write
    if "csv" in benchmark_name:
        m = re.search(r"((?:write|read)) Average", field)
        if m:
            op = m.group(1)
            dtype = "(?:int64|float64|bool|uint64|str)"
            return f"bench_{base_bench}\\[{op}-{dtype}\\]"


    # small-str-groupby
    if "small-str-groupby" in benchmark_name:
        m = re.search(r"((?:small|medium|big)) str array Average", field)
        if m:
            op = m.group(1)
            return f"bench_groupby_small_str\\[{op}\\]"


    # Sort-cases
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

    # Reduce/Scan/Aggregate
    if benchmark_name in {"reduce", "scan", "aggregate"}:
        op = field.split()[0]
        return f"bench_{benchmark_name}\\[{op}\\]"

    # Flatten (looser match)
    if benchmark_name == "flatten":
        return r"bench_flatten.*"

    # Flatten (looser match)
    if benchmark_name == "flatten":
        return r"^bench_flatten.*$"

    # Dataframe (special case)
    if benchmark_name == "dataframe":
        return r"^bench_dataframe.*$"

    # Bigint Stream (special case)
    if benchmark_name == "bigint_stream":
        return r"bench_bigint_stream\[bigint\]"

    # Default case (e.g., stream, argsort, gather, scatter)
    if benchmark_name.startswith("str(?:_|-)"):
        dtype = "str"
    elif benchmark_name.startswith("bigint(?:_|-)"):
        dtype = "bigint"
    else:
        dtype = "(?:int64|float64|bool|uint64)"

    return f"bench_{base_bench}\\[{dtype}\\]"


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
                "benchmark_name": benchmark_name.replace("str-", "")
                .replace("bigint-", "")
                .replace("-", "_"),
                "lookup_path": lookup_path,
                "lookup_regex": regex,
            }

    return field_lookup_map


def add_default_mappings(field_lookup_map):
    for b in DEFAULT_BENCHMARKS:
        if b not in field_lookup_map:
            base_bench = b.replace("str-", "").replace("bigint-", "").replace("-", "_")
            if b.startswith("str-"):
                dtype = "str"
            elif b.startswith("bigint-"):
                dtype = "bigint"
            else:
                dtype = "(?:int64|float64|bool|uint64)"

            # ✅ Special cases must come first
            if base_bench == "flatten":
                regex = r"^bench_flatten.*$"
            elif base_bench == "dataframe":
                regex = r"^bench_dataframe.*$"
            elif base_bench == "bigint_stream":
                regex = r"bench_bigint_stream\[bigint\]"
            else:
                regex = f"bench_{base_bench}\\[{dtype}\\]"

            field_lookup_map[b] = {
                "Average rate =": {
                    "group": GROUP_MAP.get(b, ""),
                    "name": f"bench_{base_bench}",
                    "benchmark_name": base_bench,
                    "lookup_path": ["extra_info", "transfer_rate"],
                    "lookup_regex": regex,
                },
                "Average time =": {
                    "group": GROUP_MAP.get(b, ""),
                    "name": f"bench_{base_bench}",
                    "benchmark_name": base_bench,
                    "lookup_path": ["stats", "mean"],
                    "lookup_regex": regex,
                },
            }
    return field_lookup_map


def add_aggregate_ops(field_lookup_map):
    if "aggregate" not in field_lookup_map:
        field_lookup_map["aggregate"] = {}
    if "reduce" not in field_lookup_map:
        field_lookup_map["reduce"] = {}

    for op in AGGREGATE_OPS:  # should include all GroupBy.Reductions ops
        for t in ["time", "rate"]:
            lookup_path = ["extra_info", "transfer_rate"] if t == "rate" else ["stats", "mean"]

            # ✅ Correct mapping for GroupBy.aggregate
            field_lookup_map["aggregate"][f"Aggregate {op} Average {t} ="] = {
                "group": "GroupBy.aggregate",
                "name": f"bench_aggregate[{op}]",
                "benchmark_name": "aggregate",
                "lookup_path": lookup_path,
                "lookup_regex": f"^bench_aggregate\\[{op}\\]$",
            }

    # ✅ Keep reduce ops separate (only numeric ops)
    for op in ["sum", "prod", "min", "max", "argmin", "argmax"]:
        for t in ["time", "rate"]:
            lookup_path = ["extra_info", "transfer_rate"] if t == "rate" else ["stats", "mean"]
            field_lookup_map["reduce"][f"Reduce {op} Average {t} ="] = {
                "group": "reduce",
                "name": f"bench_reduce[{op}]",
                "benchmark_name": "reduce",
                "lookup_path": lookup_path,
                "lookup_regex": f"bench_reduce\\[[\\w\\d]+-{op}\\]",
            }
    return field_lookup_map


def main():
    field_lookup_map = build_field_lookup_map()
    field_lookup_map = add_default_mappings(field_lookup_map)
    field_lookup_map = add_aggregate_ops(field_lookup_map)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(field_lookup_map, f, indent=2)
    print(f"✅ Updated {OUTPUT_JSON} with {len(field_lookup_map)} benchmarks.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
