import numpy as np
import arkouda as ak
import pytest

OPS = ("sum", "prod", "min", "max")
TYPES = ("int64", "float64")


# @pytest.mark.skip_if_max_rank_less_than(3)
@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_Reduce")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_ak_reduce_multidim(benchmark, op, dtype):
    if dtype in pytest.dtype:
        cfg = ak.get_config()
        N = 10**6 // 4  # (pytest.prob_size * cfg["numLocales"]) // 4
        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                a = ak.randint(1, N, 4 * N, seed=pytest.seed).reshape((2, 2, N))
            elif dtype == "float64":
                a = ak.uniform(4 * N, seed=pytest.seed).reshape((2, 2, N)) + 0.5
        else:
            a = ak.arange(4 * N).reshape((2, 2, N))
            if dtype == "float64":
                a = 1.0 * a

        fxn = getattr(a, op)
        benchmark.pedantic(fxn, rounds=pytest.trials, kwargs={"axis": 0})

        nbytes = a.size * a.itemsize
        benchmark.extra_info["description"] = "Measures performance of ak reduce functions."
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )
