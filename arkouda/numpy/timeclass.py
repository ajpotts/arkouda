
from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from pandas import Series as pdSeries

from arkouda.numpy.dtypes import int64, intTypes, isSupportedInt
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import from_series

__all__ = ["Datetime", "Timedelta"]


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
        u = u.lower()
        for key in cls:
            if u.startswith(key.name.lower()) or u == key.value:
                return key
        raise ValueError(f"Unsupported time unit: {u}")


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
        raise TypeError("Implicit array coercion for Arkouda time arrays is disabled; use .to_ndarray().")

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


class Datetime(_TimeArray):
    def _np_dtype(self) -> str:
        return "datetime64[ns]"

    def _scalar_callback(self, scalar: int) -> pd.Timestamp:
        return pd.Timestamp(int(scalar), unit="ns")

    # ----- arithmetic -----
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
        if isinstance(other, Timedelta):
            return Timedelta(self.values._binop(other.values, "%"))
        if _is_timedelta_scalar(other):
            return Timedelta(self.values._binop(normalize_td_to_ns(other), "%"))
        raise TypeError("Datetime % unsupported operand")

    def __floordiv__(self, other):
        if isinstance(other, Timedelta):
            return self.values._binop(other.values, "//")
        if _is_timedelta_scalar(other):
            return self.values._binop(normalize_td_to_ns(other), "//")
        raise TypeError("Datetime // unsupported operand")

    # ----- rounding -----
    def round(self, freq: str):
        """
        Round each timestamp to the nearest multiple of `freq` using pandas' half-even rule.
        Supported: 'ns','us','ms','s','m','h','d','w'.
        """
        unit = TimeUnit.normalize(freq)
        factor = unit.factor

        q = self.values // factor
        r = self.values % factor
        half = factor // 2

        # Start with floor
        candidate = q

        # r > half -> round up
        gt_mask = r > half
        candidate = candidate._where(gt_mask, candidate + 1)

        # r == half -> round to even (if odd, add 1)
        eq_mask = r == half
        is_odd = (candidate % 2) != 0
        add_one = is_odd._binop(eq_mask, "&")
        candidate = candidate._where(add_one, candidate + 1)

        rounded = candidate * factor
        return Datetime(rounded)

    # ----- comparisons -----
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


class Timedelta(_TimeArray):
    def _np_dtype(self) -> str:
        return "timedelta64[ns]"

    def _scalar_callback(self, scalar: int) -> pd.Timedelta:
        return pd.Timedelta(int(scalar), unit="ns")

    # ----- arithmetic -----
    def __add__(self, other):
        if isinstance(other, (Timedelta, pd.Timedelta, np.timedelta64, _dt.timedelta)):
            ns = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return Timedelta(self.values._binop(ns, "+"))
        if isinstance(other, (Datetime, pd.Timestamp, np.datetime64, _dt.datetime)):
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
        if isinstance(other, (Timedelta, pd.Timedelta, np.timedelta64, _dt.timedelta)):
            ns = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return Timedelta(self.values._binop(ns, "-"))
        raise TypeError("Timedelta - unsupported operand")

    def __rsub__(self, other):
        if _is_datetime_scalar(other):
            return Datetime(self.values._r_binop(normalize_to_ns(other), "-"))
        if _is_timedelta_scalar(other):
            return Timedelta(self.values._r_binop(normalize_td_to_ns(other), "-"))
        raise TypeError(f"unsupported operand type(s) for -: {type(other).__name__} and Timedelta")

    def __rmod__(self, other):
        if _is_datetime_scalar(other):
            return Timedelta(self.values._r_binop(normalize_to_ns(other), "%"))
        if _is_timedelta_scalar(other):
            return Timedelta(self.values._r_binop(normalize_td_to_ns(other), "%"))
        raise TypeError("unsupported operand for % with Timedelta")

    def __rfloordiv__(self, other):
        if _is_datetime_scalar(other):
            return self.values._r_binop(normalize_to_ns(other), "//")
        if isinstance(other, (int, float, np.integer, np.floating)):
            raise TypeError("Unsupported // between number and Timedelta")
        raise TypeError("unsupported operand for // with Timedelta")

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) or (isinstance(other, pdarray) and other.dtype in intTypes):
            return Timedelta(self.values._binop(other, "*"))
        raise TypeError("Timedelta * unsupported operand")

    def __rmul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) or (isinstance(other, pdarray) and other.dtype in intTypes):
            return Timedelta(self.values._r_binop(other, "*"))
        raise TypeError("unsupported * Timedelta")

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) or (isinstance(other, pdarray) and other.dtype in intTypes):
            return Timedelta(self.values._binop(other, "/"))
        if isinstance(other, (Timedelta, pd.Timedelta, np.timedelta64, _dt.timedelta)):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return self.values._binop(rhs, "/")
        raise TypeError("Timedelta / unsupported operand")

    def __rtruediv__(self, other):
        raise TypeError("unsupported / Timedelta")

    def __floordiv__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) or (isinstance(other, pdarray) and other.dtype in intTypes):
            return Timedelta(self.values._binop(other, "//"))
        if isinstance(other, (Timedelta, pd.Timedelta, np.timedelta64, _dt.timedelta)):
            rhs = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return Timedelta(self.values._binop(rhs, "//"))
        raise TypeError("Timedelta // unsupported operand")

    def __neg__(self):
        return Timedelta((-1) * self.values)

    # ----- comparisons -----
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
