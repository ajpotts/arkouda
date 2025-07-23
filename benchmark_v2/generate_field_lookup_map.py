#!/usr/bin/env python3
import os
import re
import json

GRAPH_INFRA_DIR = "benchmark_v2/graph_infra"
OUTPUT_JSON = "benchmark_v2/datdir/configs/field_lookup_map.json"

# Group inference (extend as needed)
GROUP_MAP = {
    "groupby": "GroupBy_Creation",
    # "str-groupby": "GroupBy_Creation",
    # "coargsort": "Arkouda_CoArgSort",
    # "str-coargsort": "Arkouda_CoArgSort",
    # "sort-cases": "AK_Sort_Cases",
    # "scan": "scan",
    # "reduce": "reduce",
    # "aggregate": "aggregate",
    # "stream": "stream",
    # "argsort": "argsort",
    # "gather": "gather",
    # "scatter": "scatter",
    # add others as needed

    # "stream",
# "argsort",
# "coargsort",
# "groupby",
    # "flatten",
# "aggregate",
# "gather",
# "scatter",
    # "reduce",
    # "in1d",
    # "scan",
    # "noop",
    # "setops",
    # "array_create",
    # "array_transfer",
    # "IO",
    # "csvIO",
    # "small-str-groupby",
    # "str-argsort",
    # "str-coargsort",
    # "str-groupby",
    # "str-gather",
    # "str-in1d",
    # "substring_search",
    # "split",
    # "sort-cases",
    # "multiIO",
    # "str-locality",
# "dataframe",
    # "encode",
    # "bigint_conversion",
# "bigint_stream",
# "bigint_bitwise_binops",
# "bigint_groupby",
# "bigint_array_transfer",

#   aggregate.perfkeys
#  arkouda-bigint.graph
#   arkouda-comp.graph
#   arkouda.graph
#  2744  arkouda-sort-cases.graph
#  2520  arkouda-string.graph
#   128  array_create.perfkeys
#   100  array_transfer.perfkeys
#   100  bigint_array_transfer.perfkeys
#   158  bigint_bitwise_binops.perfkeys
#   152  bigint_conversion.perfkeys
#   186  bigint_groupby.perfkeys
#    58  bigint_stream.perfkeys
#   186  coargsort.perfkeys
#    13  comp-time.perfkeys
#    97  csvIO.perfkeys
#   133  dataframe.perfkeys
#    20  emitted-code-size.perfkeys
#   219  encode.perfkeys
#   320  flatten.perfkeys
#   214  GRAPHLIST
#   186  groupby.perfkeys
#    86  in1d.perfkeys
#   101  IO.perfkeys
#   101  multiIO.perfkeys
#   624  parquetIO.perfkeys
#   624  parquetMultiIO.perfkeys
#    30  perfkeys
#   750  README.md
#   154  reduce.perfkeys
#    90  scan.perfkeys
#   197  setops.perfkeys
#   184  small-str-groupby.perfkeys
#  2426  sort-cases.perfkeys
#   186 Jul 23 16:00 str-coargsort.perfkeys
#   186  str-groupby.perfkeys
#    86  str-in1d.perfkeys
#   672  str-locality.perfkeys
#  substring_search.perfkeys


}

import argparse

import json
import logging
import os
import re

import sys


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

    parser.add_argument("--platform-name", default="", help="Test platform name")
    parser.add_argument("--description", default="", help="Description of this configuration")
    parser.add_argument("--annotations", default="", help="File containing annotations")
    parser.add_argument("--configs", help="comma seperate list of configurations")
    parser.add_argument("--start-date", help="graph start date")
    parser.add_argument("--benchmark-data", help="the benchnmark output data in json format.")

    return parser

def infer_regex(benchmark_name: str, field: str) -> str:
    """Infer a regex for JSON benchmark names based on perfkey field names."""
    base_bench = benchmark_name.replace("str-", "").replace("bigint-", "")

    # Handle array cases (groupby, coargsort, etc.)
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



def gen_lookup_map(write=False, out_file="field_lookup_map.json"):
    """Temporarily use a script to generate the lookup dictionary and save to file when write=True."""
    field_lookup_map = {}
    for benchmark_name in BENCHMARKS:
        field_lookup_map[benchmark_name] = {}

        field_lookup_map[benchmark_name]["Average rate ="] = get_lookup_dict(
            name="bench_" + benchmark_name,
            benchmark_name=benchmark_name,
            lookup_path=[
                "extra_info",
                "transfer_rate",
            ],
            lookup_regex="bench_" + benchmark_name + r"\[[\w\d]*\]",
        )

        field_lookup_map[benchmark_name]["Average time ="] = get_lookup_dict(
            name="bench_" + benchmark_name,
            benchmark_name=benchmark_name,
            lookup_path=[
                "stats",
                "mean",
            ],
            lookup_regex="bench_" + benchmark_name + r"\[[\w\d]*\]",
        )

    for op in [
        "prod",
        "sum",
        "mean",
        "min",
        "max",
        "argmin",
        "argmax",
        "any",
        "all",
        "xor",
        "and",
        "or",
        "nunique",
    ]:
        field_lookup_map["aggregate"][f"Aggregate {op} Average rate ="] = get_lookup_dict(
            group="GroupBy.aggregate",
            name=f"bench_aggregate[{op}]",
            benchmark_name="aggregate",
            lookup_path=[
                "extra_info",
                "transfer_rate",
            ],
        )

        field_lookup_map["aggregate"][f"Aggregate {op} Average time ="] = get_lookup_dict(
            group="GroupBy.aggregate",
            name=f"bench_aggregate[{op}]",
            benchmark_name="aggregate",
            lookup_path=[
                "stats",
                "mean",
            ],
        )

    for num in [1, 2, 8, 16]:
        for key, (group, bench) in {
            "coargsort": ("Arkouda_CoArgSort", "coargsort"),
            "groupby": ("GroupBy_Creation", "groupby"),
        }.items():
            regex = f"bench_{bench}\\[[\\w\\d]*-{num}\\]"

            field_lookup_map[key][f"{num}-array Average rate ="] = get_lookup_dict(
                group=group,
                benchmark_name=bench,
                lookup_path=["extra_info", "transfer_rate"],
                lookup_regex=regex,
            )

            field_lookup_map[key][f"{num}-array Average time ="] = get_lookup_dict(
                group=group,
                benchmark_name=bench,
                lookup_path=["stats", "mean"],
                lookup_regex=regex,
            )

    field_lookup_map["bigint_stream"]["Average bigint stream time ="] = get_lookup_dict(
        group="stream",
        name="bench_bigint_stream",
        benchmark_name="bigint_stream",
        lookup_path=[
            "stats",
            "mean",
        ],
        lookup_regex="bench_bigint_stream\\[[\\w\\d]*\\]",
    )

    #   Field aliases:
    #   bigint
    field_lookup_map["bigint_stream"]["Average bigint stream time ="] = field_lookup_map[
        "bigint_stream"
    ]["Average time ="]
    field_lookup_map["bigint_stream"]["Average bigint stream rate ="] = field_lookup_map[
        "bigint_stream"
    ]["Average rate ="]

    #   dataframe
    field_lookup_map["dataframe"]["_get_head_tail_server Average time ="] = get_lookup_dict(
        group="Dataframe_Indexing",
        name="bench_dataframe[_get_head_tail_server]",
        benchmark_name="dataframe",
        lookup_path=[
            "stats",
            "mean",
        ],
    )

    field_lookup_map["dataframe"]["_get_head_tail_server Average rate ="] = get_lookup_dict(
        group="Dataframe_Indexing",
        name="bench_dataframe[_get_head_tail_server]",
        benchmark_name="dataframe",
        lookup_path=[
            "extra_info",
            "transfer_rate",
        ],
    )

    field_lookup_map["dataframe"]["_get_head_tail Average time ="] = get_lookup_dict(
        group="Dataframe_Indexing",
        name="bench_dataframe[_get_head_tail]",
        benchmark_name="dataframe",
        lookup_path=[
            "stats",
            "mean",
        ],
    )

    field_lookup_map["dataframe"]["_get_head_tail Average rate ="] = get_lookup_dict(
        group="Dataframe_Indexing",
        name="bench_dataframe[_get_head_tail]",
        benchmark_name="dataframe",
        lookup_path=[
            "extra_info",
            "transfer_rate",
        ],
    )

    if write:
        with open(out_file, "w") as fp:
            json.dump(field_lookup_map, fp)

    return field_lookup_map


def get_lookup_dict(group="", name="", benchmark_name="", lookup_path=[], lookup_regex=""):
    """Populate the lookup dictionary fields and return a dictionary."""
    ret_dict = {
        "group": group,
        "name": name,
        "benchmark_name": benchmark_name,
        "lookup_path": lookup_path,
        "lookup_regex": lookup_regex,
    }
    return ret_dict

def build_field_lookup_map():
    headers = get_header_fields_from_directory(GRAPH_INFRA_DIR)
    print(headers)
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


    field_lookup_map = build_field_lookup_map()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(field_lookup_map, f, indent=2)
    print(f"Updated {OUTPUT_JSON} with {len(field_lookup_map)} benchmarks.")


if __name__ == "__main__":
    main()

