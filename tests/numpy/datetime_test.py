import numpy as np
import pandas as pd
import pytest
import arkouda as ak

# -------------------------------------------------------------
# Helpers: normalize results to comparable numpy arrays
# -------------------------------------------------------------

_OPS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "//": lambda x, y: x // y,
    "%": lambda x, y: x % y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
}

def _pandas_try(op, left, right):
    try:
        res = _OPS[op](left, right)
        return True, res
    except Exception as e:
        return False, e

def _as_ns64(arr):
    """Return numpy int64 nanoseconds for datetime/timedelta arrays; otherwise original values."""
    a = np.asarray(arr)
    # Try datetime first
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype('datetime64[ns]').astype('int64', copy=False)
    # Then timedelta
    if np.issubdtype(a.dtype, np.timedelta64):
        return a.astype('timedelta64[ns]').astype('int64', copy=False)
    # Object array: try to coerce via pandas
    if a.dtype == object:
        try:
            b = pd.to_datetime(a, errors='raise').values.astype('datetime64[ns]').astype('int64', copy=False)
            return b
        except Exception:
            try:
                b = pd.to_timedelta(a, errors='raise').values.astype('timedelta64[ns]').astype('int64', copy=False)
                return b
            except Exception:
                return a
    return a

def _ak_nd(x):
    if hasattr(x, 'to_ndarray'):
        return x.to_ndarray()
    if hasattr(x, 'dtype'):
        return x.to_ndarray()
    return np.asarray(x)

# -------------------------------------------------------------
# NOTE: No connection code here. Your conftest.py manages it.
# -------------------------------------------------------------

@pytest.fixture(scope="function")
def data():
    dt_vec = ak.date_range(start="2021-01-01 12:00:00", periods=100, freq="s")
    td_vec = ak.timedelta_range(start=0, periods=100, freq="s")
    int_vec = ak.arange(100)

    dt_scalar = pd.Timestamp("2021-01-01 12:00:00")
    td_scalar = pd.Timedelta(1, unit="s")
    num_scalar = 5

    pd_dt_vec = pd.to_datetime(dt_vec.to_ndarray())
    pd_td_vec = pd.to_timedelta(td_vec.to_ndarray())
    pd_int_vec = pd.Series(int_vec.to_ndarray())

    return {
        "ak": {
            "Datetime": dt_vec,
            "Timedelta": td_vec,
            "pdarray": int_vec,
            "dt_scalar": dt_scalar,
            "td_scalar": td_scalar,
            "num_scalar": num_scalar,
        },
        "pd": {
            "Datetime": pd_dt_vec,
            "Timedelta": pd_td_vec,
            "pdarray": pd_int_vec,
            "dt_scalar": dt_scalar,
            "td_scalar": td_scalar,
            "num_scalar": num_scalar,
        },
    }

class TestDatetimePandasAligned:
    def test_creation_equivalence(self):
        dt = ak.date_range(start="2021-01-01", periods=3, freq="s")
        pd_dt = pd.date_range(start="2021-01-01", periods=3, freq="s")
        np.testing.assert_equal(_as_ns64(pd_dt.values), _as_ns64(dt.to_ndarray()))

        td = ak.timedelta_range(start=0, periods=3, freq="s")
        pd_td = pd.to_timedelta(np.arange(3), unit="s")
        np.testing.assert_equal(_as_ns64(pd_td.values), _as_ns64(td.to_ndarray()))

    @pytest.mark.parametrize("op", list(_OPS.keys()))
    def test_ops_vector_scalar_and_scalar_vector(self, data, op):
        ak_objs = data["ak"]
        pd_objs = data["pd"]

        LEFT = (("Datetime","dt_scalar"), ("Datetime","td_scalar"), ("Datetime","num_scalar"),
                ("Timedelta","dt_scalar"), ("Timedelta","td_scalar"), ("Timedelta","num_scalar"))

        for left_kind, right_scalar_name in LEFT:
            ak_left = ak_objs[left_kind]
            pd_left = pd_objs[left_kind]

            ak_right_sc = ak_objs[right_scalar_name]
            pd_right_sc = pd_objs[right_scalar_name]

            # vector <op> scalar
            pd_supported, pd_res = _pandas_try(op, pd_left, pd_right_sc)
            if pd_supported:
                ak_res = _OPS[op](ak_left, ak_right_sc)
                # basic type sanity (kept minimal)
                if left_kind == "Datetime":
                    if op in {"+", "-"} and right_scalar_name == "td_scalar":
                        assert isinstance(ak_res, ak.Datetime)
                    elif op == "-" and right_scalar_name == "dt_scalar":
                        assert isinstance(ak_res, ak.Timedelta)
                if left_kind == "Timedelta":
                    if op in {"+","-"} and right_scalar_name in {"td_scalar","num_scalar"}:
                        assert isinstance(ak_res, ak.Timedelta)
                np.testing.assert_equal(_as_ns64(pd_res), _as_ns64(_ak_nd(ak_res)))
            else:
                with pytest.raises(TypeError):
                    _OPS[op](ak_left, ak_objs[right_scalar_name])

            # scalar <op> vector (reflected)
            pd_supported_r, pd_res_r = _pandas_try(op, pd_right_sc, pd_left)
            if pd_supported_r:
                ak_res_r = _OPS[op](ak_objs[right_scalar_name], ak_left)
                np.testing.assert_equal(_as_ns64(pd_res_r), _as_ns64(_ak_nd(ak_res_r)))
            else:
                with pytest.raises(TypeError):
                    _OPS[op](ak_objs[right_scalar_name], ak_left)

    @pytest.mark.parametrize("op", ["+","-","==","!=", "<","<=",">",">=","/","//","%"])
    def test_ops_vector_vector(self, data, op):
        ak_objs = data["ak"]
        pd_objs = data["pd"]

        PAIRS = (("Datetime","Datetime"), ("Datetime","Timedelta"),
                 ("Timedelta","Datetime"), ("Timedelta","Timedelta"))

        for left_kind, right_kind in PAIRS:
            ak_left, ak_right = ak_objs[left_kind], ak_objs[right_kind]
            pd_left, pd_right = pd_objs[left_kind], pd_objs[right_kind]

            pd_supported, pd_res = _pandas_try(op, pd_left, pd_right)
            if pd_supported:
                ak_res = _OPS[op](ak_left, ak_right)
                np.testing.assert_equal(_as_ns64(pd_res), _as_ns64(_ak_nd(ak_res)))
            else:
                with pytest.raises(TypeError):
                    _OPS[op](ak_left, ak_right)

    def test_round_to_minute(self):
        vec = ak.date_range(start="2021-01-01 12:00:00", periods=100, freq="s")
        rounded = vec.round("m")
        np.testing.assert_equal(
            _as_ns64(pd.to_datetime(vec.to_ndarray()).round("min").values),
            _as_ns64(rounded.to_ndarray()),
        )

    def test_date_time_attribute_accessors(self):
        ak_dt = ak.Datetime(ak.date_range("2021-01-01 00:00:00", periods=100))
        pd_dt = pd.Series(pd.date_range("2021-01-01 00:00:00", periods=100)).dt

        assert (pd_dt.date == ak_dt.date.to_ndarray()).all()
        for attr in ("nanosecond","microsecond","second","minute","hour","day","month","year",
                     "day_of_week","dayofweek","weekday","day_of_year","dayofyear","is_leap_year"):
            assert getattr(pd_dt, attr).tolist() == getattr(ak_dt, attr).tolist()

        assert pd_dt.isocalendar().week.tolist() == ak_dt.week.tolist()
        assert pd_dt.isocalendar().week.tolist() == ak_dt.weekofyear.tolist()
        ak_iso = pd.DataFrame({
            'year': ak_dt.isocalendar().year.to_ndarray(),
            'week': ak_dt.isocalendar().week.to_ndarray(),
            'day': ak_dt.isocalendar().day.to_ndarray(),
        })
        assert ((pd_dt.isocalendar() == ak_iso).all()).all()

    def test_timedelta_accessors_and_stats(self):
        ak_td = ak.Timedelta(ak.arange(10**6, 10**6 + 1000), unit="us")
        pd_td = pd.Series(pd.to_timedelta(np.arange(10**6, 10**6 + 1000), unit="us")).dt

        ak_comp = pd.DataFrame({k: getattr(ak_td.components, k).to_ndarray()
                                for k in ('days','hours','minutes','seconds','milliseconds','microseconds','nanoseconds')})
        assert ((pd_td.components == ak_comp).all()).all()
        assert np.allclose(pd_td.total_seconds(), ak_td.total_seconds().to_ndarray())
        for attr in ("nanoseconds","microseconds","seconds","days"):
            assert getattr(pd_td, attr).tolist() == getattr(ak_td, attr).tolist()

        ak_std = ak.Timedelta(ak.array([123,456,789]), unit="s").std(ddof=1)
        pd_std = pd.to_timedelta([123,456,789], unit="s").std()
        assert ak_std == pd_std

    def test_min_max_and_sum(self):
        dt = ak.date_range(start="2021-01-01 00:00:00", periods=10, freq="h")
        assert dt.min() == dt[0]
        assert dt.max() == dt[-1]
        with pytest.raises(TypeError):
            _ = dt.sum()

        td = ak.timedelta_range(start=0, periods=10, freq="s")
        assert td.min() == pd.Timedelta(0, unit="s")
        assert td.max() == pd.Timedelta(9, unit="s")
        assert td.sum() == pd.Timedelta(td.size - 1, unit="s")
        assert ((-td).abs() == td).all()

    def test_unit_aliases(self):
        unitmap = {
            "W": ("weeks", "w", "week"),
            "D": ("days", "d", "day"),
            "h": ("hours", "H", "hr", "hrs"),
            "m": ("minutes", "minute", "min", "m"),
            "ms": ("milliseconds", "millisecond", "milli", "ms", "l"),
            "us": ("microseconds", "microsecond", "micro", "us", "u"),
            "ns": ("nanoseconds", "nanosecond", "nano", "ns", "n"),
        }
        for pdunit, aliases in unitmap.items():
            for akunit in (pdunit,) + aliases:
                for pdclass, akclass in ((pd.Timestamp, ak.Datetime), (pd.Timedelta, ak.Timedelta)):
                    pdval = pdclass(1, unit=pdunit)
                    akval = akclass(ak.ones(3, dtype=ak.int64), unit=akunit)[0]
                    assert pdval == akval
