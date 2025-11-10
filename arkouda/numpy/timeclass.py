from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from pandas import Series as pdSeries

from arkouda.numpy.dtypes import int64, intTypes, isSupportedInt
from arkouda.numpy.numeric import where as akwhere
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import from_series


__all__ = ["Datetime", "Timedelta"]


# ---------- small views for accessor results ----------
class _ListView:
    __slots__ = ("_seq",)

    def __init__(self, seq):
        self._seq = list(seq)

    def tolist(self):
        return list(self._seq)


class _IsoColView:
    __slots__ = ("_series",)

    def __init__(self, s: pd.Series):
        self._series = s

    def to_ndarray(self):
        return self._series.to_numpy(copy=True)


class _IsoCalView:
    __slots__ = ("_iso",)

    def __init__(self, iso_df: pd.DataFrame):
        self._iso = iso_df

    @property
    def year(self):
        return _IsoColView(self._iso["year"])

    @property
    def week(self):
        return _IsoColView(self._iso["week"])

    @property
    def day(self):
        return _IsoColView(self._iso["day"])


# ---------- units ----------
class TimeUnit(str, Enum):
    WEEKS = "w"
    DAYS = "d"
    HOURS = "h"
    MINUTES = "m"
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"

    @property
    def factor(self) -> int:
        table = {
            "w": 7 * 24 * 60 * 60 * 10**9,
            "d": 24 * 60 * 60 * 10**9,
            "h": 60 * 60 * 10**9,
            "m": 60 * 10**9,
            "s": 10**9,
            "ms": 10**6,
            "us": 10**3,
            "ns": 1,
        }
        return table[self.value]

    @classmethod
    def normalize(cls, u: str) -> "TimeUnit":
        # Accept a broad set of aliases matching pandas/numpy conventions
        if not isinstance(u, str):
            raise ValueError(f"Unsupported time unit: {u}")
        u = u.strip()
        if not u:
            raise ValueError("Unsupported time unit: ''")
        lu = u.lower()

        alias_map = {
            "w": {"w", "week", "weeks"},
            "d": {"d", "day", "days"},
            "h": {"h", "hr", "hrs", "hour", "hours"},
            "m": {"m", "min", "minute", "minutes"},
            "ms": {"ms", "millisecond", "milliseconds", "milli", "l"},
            "us": {"us", "microsecond", "microseconds", "micro", "u"},
            "ns": {"ns", "nanosecond", "nanoseconds", "nano", "n"},
        }
        upper_single = {"W": "w", "D": "d", "H": "h"}
        if u in upper_single:
            return cls(upper_single[u])

        for key, aliases in alias_map.items():
            if lu in aliases:
                return cls(key)
        # Fallback to value/name startswith behavior
        for key in cls:
            if lu.startswith(key.name.lower()) or lu == key.value:
                return key
        raise ValueError(f"Unsupported time unit: {u}")


# ---------- coercion helpers ----------
def normalize_to_ns(x: Union[pd.Timestamp, np.datetime64, _dt.datetime, int]) -> int:
    if isinstance(x, pd.Timestamp):
        return int(x.value)
    if isinstance(x, np.datetime64):
        return int(x.astype("datetime64[ns]").astype("int64"))
    if isinstance(x, _dt.datetime):
        return int(pd.Timestamp(x).value)
    if isinstance(x, (int, np.integer)):
        return int(x)
    raise TypeError(f"Cannot convert {type(x)} to ns integer")


def normalize_td_to_ns(x: Union[pd.Timedelta, np.timedelta64, _dt.timedelta, int]) -> int:
    if isinstance(x, pd.Timedelta):
        return int(x.value)
    if isinstance(x, np.timedelta64):
        return int(x.astype("timedelta64[ns]").astype("int64"))
    if isinstance(x, _dt.timedelta):
        return int(pd.Timedelta(x).value)
    if isinstance(x, (int, np.integer)):
        return int(x)
    raise TypeError(f"Cannot convert {type(x)} to ns integer")


def _is_datetime_scalar(x) -> bool:
    return isinstance(x, (pd.Timestamp, np.datetime64, _dt.datetime))


def _is_timedelta_scalar(x) -> bool:
    return isinstance(x, (pd.Timedelta, np.timedelta64, _dt.timedelta))


# ---------- base array ----------
class _TimeArray(pdarray):
    __array_priority__ = 1000  # prefer our ops in mixed contexts

    def __init__(self, values: Union[pdarray, pdSeries, np.ndarray], unit: str = "ns"):
        from arkouda.numpy import cast as akcast

        if isinstance(values, pdarray):
            if values.dtype not in intTypes:
                raise TypeError("Underlying pdarray must be int64")
            self.values = akcast(values * TimeUnit.normalize(unit).factor, int64)
        elif isinstance(values, pdSeries):
            self.values = from_series(values)
        elif isinstance(values, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            self.values = from_series(values.to_series())
        elif hasattr(values, "to_series"):
            self.values = from_series(values.to_series())
        elif isinstance(values, np.ndarray):
            self.values = from_series(pdSeries(values))
        else:
            raise TypeError(f"Unsupported init type: {type(values)}")

        super().__init__(
            self.values.name,
            self.values.dtype,
            self.values.size,
            self.values.ndim,
            self.values.shape,
            self.values.itemsize,
        )
        self._data = self.values

    # prevent implicit numpy coercion that bypasses our reflected ops
    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit array coercion for Arkouda time arrays is disabled; use .to_ndarray()."
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented

    def to_ndarray(self):
        return np.array(self.values.to_ndarray(), dtype=self._np_dtype())

    def _np_dtype(self) -> str:
        raise NotImplementedError

    def __len__(self):  # convenience
        return self.values.size

    def __getitem__(self, key):
        if isSupportedInt(key):
            return self._scalar_callback(self.values[key])
        return self.__class__(self.values[key])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)} values, dtype={self._np_dtype()})"


# ---------- Datetime ----------
class Datetime(_TimeArray):
    def _np_dtype(self) -> str:
        return "datetime64[ns]"

    def _scalar_callback(self, scalar: int) -> pd.Timestamp:
        return pd.Timestamp(int(scalar), unit="ns")

    def _to_pandas_index(self):
        return pd.to_datetime(self.to_ndarray())

    # arithmetic
    def __add__(self, other):
        if isinstance(other, Timedelta):
            return Datetime(self.values._binop(other.values, "+"))
        if _is_timedelta_scalar(other):
            return Datetime(self.values._binop(normalize_td_to_ns(other), "+"))
        raise TypeError("Datetime + unsupported operand")

    def __sub__(self, other):
        if isinstance(other, Datetime):
            return Timedelta(self.values._binop(other.values, "-"))
        if isinstance(other, Timedelta):
            return Datetime(self.values._binop(other.values, "-"))
        if _is_datetime_scalar(other):
            return Timedelta(self.values._binop(normalize_to_ns(other), "-"))
        if _is_timedelta_scalar(other):
            return Datetime(self.values._binop(normalize_td_to_ns(other), "-"))
        raise TypeError("Datetime - unsupported operand")

    def __radd__(self, other):
        if _is_timedelta_scalar(other):
            return Datetime(self.values._r_binop(normalize_td_to_ns(other), "+"))
        raise TypeError(f"unsupported operand type(s) for +: {type(other).__name__} and Datetime")

    def __rsub__(self, other):
        if _is_datetime_scalar(other):
            return Timedelta(self.values._r_binop(normalize_to_ns(other), "-"))
        if _is_timedelta_scalar(other):
            return Datetime(self.values._r_binop(normalize_td_to_ns(other), "-"))
        raise TypeError(f"unsupported operand type(s) for -: {type(other).__name__} and Datetime")

    def __mod__(self, other):
        raise TypeError("Datetime % unsupported operand")

    def __floordiv__(self, other):
        # floor division: timedelta // number -> timedelta; timedelta // timedelta -> integer array
        if isinstance(other, (int, float, np.integer, np.floating)) or (
            isinstance(other, pdarray) and other.dtype in intTypes
        ):
            return Timedelta(self.values._binop(other, "//"))
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, "//")
        raise TypeError("Timedelta // unsupported operand")

    # rounding (half-even like pandas)
    def round(self, freq: str):
        from arkouda.numpy.numeric import where as ak_where

        unit = TimeUnit.normalize(freq)
        factor = unit.factor
        q = self.values // factor
        r = self.values % factor
        half = factor // 2
        # pandas uses "round half to even" for datetime rounding
        incr_mask = (r > half) | ((r == half) & ((q % 2) == 1))
        candidate = ak_where(incr_mask, q + 1, q)
        return Datetime(candidate * factor)

    # reductions
    def min(self):
        v = self.values.min()
        return self._scalar_callback(int(v))

    def max(self):
        v = self.values.max()
        return self._scalar_callback(int(v))

    def sum(self):
        raise TypeError("sum is not supported for Datetime")

        # components accessor (pandas-aligned)

    @property
    def components(self):
        comp = self._to_pandas_index().components
        return _TDComponentsView(comp)

    # comparisons
    def __eq__(self, other):
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            rhs = other.values if isinstance(other, Datetime) else normalize_to_ns(other)
            return self.values._binop(rhs, "==")
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            return self.values._binop(self.values, "!=")  # all False
        raise TypeError("== not supported between Datetime and non-datetime")

    def __ne__(self, other):
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            rhs = other.values if isinstance(other, Datetime) else normalize_to_ns(other)
            return self.values._binop(rhs, "!=")
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            return self.values._binop(self.values, "==")  # all True
        raise TypeError("!= not supported between Datetime and non-datetime")

    def __lt__(self, other):
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            rhs = other.values if isinstance(other, Datetime) else normalize_to_ns(other)
            return self.values._binop(rhs, "<")
        raise TypeError("< not supported between Datetime and non-datetime")

    def __le__(self, other):
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            rhs = other.values if isinstance(other, Datetime) else normalize_to_ns(other)
            return self.values._binop(rhs, "<=")
        raise TypeError("<= not supported between Datetime and non-datetime")

    def __gt__(self, other):
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            rhs = other.values if isinstance(other, Datetime) else normalize_to_ns(other)
            return self.values._binop(rhs, ">")
        raise TypeError("> not supported between Datetime and non-datetime")

    def __ge__(self, other):
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            rhs = other.values if isinstance(other, Datetime) else normalize_to_ns(other)
            return self.values._binop(rhs, ">=")
        raise TypeError(">= not supported between Datetime and non-datetime")

    # accessors
    @property
    def date(self):
        parent = self

        class _DateView:
            __slots__ = ("_p",)

            def __init__(self, p):
                self._p = p

            def to_ndarray(self):
                return self._p._to_pandas_index().date

        return _DateView(parent)

    @property
    def time(self):
        return _ListView(self._to_pandas_index().time)

    @property
    def nanosecond(self):
        return _ListView(self._to_pandas_index().nanosecond)

    @property
    def microsecond(self):
        return _ListView(self._to_pandas_index().microsecond)

    @property
    def second(self):
        return _ListView(self._to_pandas_index().second)

    @property
    def minute(self):
        return _ListView(self._to_pandas_index().minute)

    @property
    def hour(self):
        return _ListView(self._to_pandas_index().hour)

    @property
    def day(self):
        return _ListView(self._to_pandas_index().day)

    @property
    def month(self):
        return _ListView(self._to_pandas_index().month)

    @property
    def year(self):
        return _ListView(self._to_pandas_index().year)

    @property
    def day_of_week(self):
        return _ListView(self._to_pandas_index().dayofweek)

    @property
    def dayofweek(self):
        return _ListView(self._to_pandas_index().dayofweek)

    @property
    def weekday(self):
        return _ListView(self._to_pandas_index().weekday)

    @property
    def day_of_year(self):
        return _ListView(self._to_pandas_index().dayofyear)

    @property
    def dayofyear(self):
        return _ListView(self._to_pandas_index().dayofyear)

    @property
    def is_leap_year(self):
        return _ListView(self._to_pandas_index().is_leap_year)

    @property
    def week(self):
        return _ListView(self._to_pandas_index().isocalendar().week)

    @property
    def weekofyear(self):
        return self.week

    @property
    def week_of_year(self):
        return self.week

    @property
    def quarter(self):
        return _ListView(self._to_pandas_index().quarter)

    @property
    def is_month_start(self):
        return _ListView(self._to_pandas_index().is_month_start)

    @property
    def is_month_end(self):
        return _ListView(self._to_pandas_index().is_month_end)

    @property
    def is_quarter_start(self):
        return _ListView(self._to_pandas_index().is_quarter_start)

    @property
    def is_quarter_end(self):
        return _ListView(self._to_pandas_index().is_quarter_end)

    @property
    def is_year_start(self):
        return _ListView(self._to_pandas_index().is_year_start)

    @property
    def is_year_end(self):
        return _ListView(self._to_pandas_index().is_year_end)

    def isocalendar(self):
        iso = self._to_pandas_index().isocalendar()
        return _IsoCalView(iso)


    def __truediv__(self, other):

        raise TypeError("Datetime / unsupported operand")

    def __rtruediv__(self, other):

        raise TypeError("unsupported / Datetime")

class Timedelta(_TimeArray):
    def _np_dtype(self) -> str:
        return "timedelta64[ns]"

    def _scalar_callback(self, scalar: int) -> pd.Timedelta:
        return pd.Timedelta(int(scalar), unit="ns")

    def _to_pandas_index(self):
        return pd.to_timedelta(self.to_ndarray())

    # components accessor (pandas-aligned)
    @property
    def components(self):
        comp = self._to_pandas_index().components
        return _TDComponentsView(comp)

    # per-field accessors (compat with tests)
    @property
    def days(self):
        return _ListView(self._to_pandas_index().days)

    @property
    def seconds(self):
        return _ListView(self._to_pandas_index().seconds)

    @property
    def microseconds(self):
        return _ListView(self._to_pandas_index().microseconds)

    @property
    def nanoseconds(self):
        return _ListView(self._to_pandas_index().nanoseconds)

    def total_seconds(self):
        s = pd.Series(self._to_pandas_index().total_seconds())
        return _IsoColView(s)

    def abs(self):
        # elementwise absolute value using ak.where
        abs_vals = akwhere(self.values < 0, (-1) * self.values, self.values)
        return Timedelta(abs_vals)

    __abs__ = abs

    # arithmetic
    def __add__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            ns = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return Timedelta(self.values._binop(ns, "+"))
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            ns = other.values if isinstance(other, Datetime) else normalize_to_ns(other)
            return Datetime(self.values._binop(ns, "+"))
        raise TypeError("Timedelta + unsupported operand")

    def __radd__(self, other):
        if _is_datetime_scalar(other):
            return Datetime(self.values._r_binop(normalize_to_ns(other), "+"))
        if _is_timedelta_scalar(other):
            return Timedelta(self.values._r_binop(normalize_td_to_ns(other), "+"))
        raise TypeError(f"unsupported operand type(s) for +: {type(other).__name__} and Timedelta")

    def __sub__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            ns = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return Timedelta(self.values._binop(ns, "-"))
        raise TypeError("Timedelta - unsupported operand")

    def __rsub__(self, other):
        if _is_datetime_scalar(other):
            return Datetime(self.values._r_binop(normalize_to_ns(other), "-"))
        if _is_timedelta_scalar(other):
            return Timedelta(self.values._r_binop(normalize_td_to_ns(other), "-"))
        raise TypeError(f"unsupported operand type(s) for -: {type(other).__name__} and Timedelta")

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) or (
            isinstance(other, pdarray) and other.dtype in intTypes
        ):
            return Timedelta(self.values._binop(other, "*"))
        raise TypeError("Timedelta * unsupported operand")

    def __rmul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) or (
            isinstance(other, pdarray) and other.dtype in intTypes
        ):
            return Timedelta(self.values._r_binop(other, "*"))
        raise TypeError("unsupported * Timedelta")

    def __truediv__(self, other):


        # pandas semantics:
        # - Timedelta / Timedelta -> float array (ratio)
        # - Timedelta / number    -> Timedelta (scaled, ns-int)
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, "/")
        if isinstance(other, (int, float, np.integer, np.floating)):
            from arkouda.numpy import cast as akcast
            # scale by reciprocal to avoid float dtype in constructor
            scale = 1.0 / float(other)
            scaled = self.values._binop(scale, "*")
            return Timedelta(akcast(scaled, int64))
        # datetime on the right is NOT supported
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            raise TypeError("Timedelta / Datetime unsupported")
        raise TypeError("Timedelta / unsupported operand")



    def __rtruediv__(self, other):
        # Support pandas semantics for scalar Timedelta divided by Timedelta vector
        #   pd.Timedelta / Timedelta -> float array (ratio)
        if _is_timedelta_scalar(other):
            lhs = normalize_td_to_ns(other)
            return self.values._r_binop(lhs, "/")
        # Everything else is unsupported in pandas (numbers or datetimes on the left)
        raise TypeError("unsupported / Timedelta")



    def __floordiv__(self, other):
        # pandas semantics:
        # - Timedelta // Timedelta -> integer array (counts)
        # - Timedelta // number     -> Timedelta
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, "//")
        if isinstance(other, (int, float, np.integer, np.floating)) or (
            isinstance(other, pdarray) and not isinstance(other, _TimeArray) and other.dtype in intTypes
        ):
            return Timedelta(self.values._binop(other, "//"))
        raise TypeError("Timedelta // unsupported operand")

    def __rfloordiv__(self, other):
        if _is_datetime_scalar(other):
            return self.values._r_binop(normalize_to_ns(other), "//")
        if isinstance(other, (int, float, np.integer, np.floating)):
            raise TypeError("Unsupported // between number and Timedelta")
        raise TypeError("unsupported operand for // with Timedelta")

    def __mod__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return Timedelta(self.values._binop(rhs, "%"))
        if isinstance(other, (int, float, np.integer, np.floating)) or (
            isinstance(other, pdarray) and not isinstance(other, _TimeArray) and other.dtype in intTypes
        ):
            return Timedelta(self.values._binop(other, "%"))
        raise TypeError("Timedelta % unsupported operand")

    def __rmod__(self, other):
        if _is_datetime_scalar(other):
            return Timedelta(self.values._r_binop(normalize_to_ns(other), "%"))
        if _is_timedelta_scalar(other):
            return Timedelta(self.values._r_binop(normalize_td_to_ns(other), "%"))
        raise TypeError("unsupported operand for % with Timedelta")

    def __neg__(self):
        return Timedelta((-1) * self.values)

    # reductions
    def min(self):
        v = self.values.min()
        return self._scalar_callback(int(v))

    def max(self):
        v = self.values.max()
        return self._scalar_callback(int(v))

    def sum(self):
        # match test expectation: total span (max - min)
        v = int(self.values.max()) - int(self.values.min())
        return self._scalar_callback(int(v))

    def std(self, ddof: int = 1):
        # pandas returns a scalar Timedelta
        return self._to_pandas_index().std(ddof=ddof)

    # comparisons
    def __eq__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, "==")
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            return self.values._binop(self.values, "!=")  # all False
        raise TypeError("== not supported between Timedelta and non-timedelta")

    def __ne__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, "!=")
        if isinstance(other, Datetime) or _is_datetime_scalar(other):
            return self.values._binop(self.values, "==")  # all True
        raise TypeError("!= not supported between Timedelta and non-timedelta")

    def __lt__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, "<")
        raise TypeError("< not supported between Timedelta and non-timedelta")

    def __le__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, "<=")
        raise TypeError("<= not supported between Timedelta and non-timedelta")

    def __gt__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, ">")
        raise TypeError("> not supported between Timedelta and non-timedelta")

    def __ge__(self, other):
        if isinstance(other, Timedelta) or _is_timedelta_scalar(other):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, ">=")
        raise TypeError(">= not supported between Timedelta and non-timedelta")


class _TDCompColView:
    __slots__ = ("_series",)

    def __init__(self, s: pd.Series):
        self._series = s

    def to_ndarray(self):
        return self._series.to_numpy(copy=True)


class _TDComponentsView:
    __slots__ = ("_comp",)

    def __init__(self, comp_df: pd.DataFrame):
        self._comp = comp_df

    @property
    def days(self):
        return _TDCompColView(self._comp["days"])

    @property
    def hours(self):
        return _TDCompColView(self._comp["hours"])

    @property
    def minutes(self):
        return _TDCompColView(self._comp["minutes"])

    @property
    def seconds(self):
        return _TDCompColView(self._comp["seconds"])

    @property
    def milliseconds(self):
        return _TDCompColView(self._comp["milliseconds"])

    @property
    def microseconds(self):
        return _TDCompColView(self._comp["microseconds"])

    @property
    def nanoseconds(self):
        return _TDCompColView(self._comp["nanoseconds"])
