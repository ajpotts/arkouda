# improved_timeclass.py
# ---------------------------------------------------------------------------
# Simplified, maintainable rewrite of Arkouda's timeclass.py
# ---------------------------------------------------------------------------
from __future__ import annotations

import datetime as _dt
from enum import Enum
import json
import numbers
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas import Series as pdSeries
from pandas import Timedelta as pdTimedelta
from pandas import Timestamp as pdTimestamp

from arkouda.numpy.dtypes import int64, intTypes, isSupportedInt
from arkouda.numpy.pdarrayclass import RegistrationError, create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import from_series


__all__ = ["Datetime", "Timedelta"]


class TimeUnit(str, Enum):
    """Enumeration of supported time units (nanoseconds base)."""

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
    """Convert any datetime-like to integer nanoseconds."""
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


class _TimeArray(pdarray):
    """Base class for Datetime and Timedelta with nanosecond precision."""

    def __init__(self, values: Union[pdarray, pdSeries, np.ndarray], unit: str = "ns"):
        from arkouda.numpy import cast as akcast

        if isinstance(values, pdarray):
            if values.dtype not in intTypes:
                raise TypeError("Underlying pdarray must be int64")
            self.values = akcast(values * TimeUnit.normalize(unit).factor, int64)
        elif isinstance(values, pdSeries):
            self.values = from_series(values)
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

    def to_ndarray(self):
        return np.array(self.values.to_ndarray(), dtype=self._np_dtype())

    def _np_dtype(self) -> str:
        raise NotImplementedError

    def __getitem__(self, key):
        if isSupportedInt(key):
            return self._scalar_callback(self.values[key])
        return self.__class__(self.values[key])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)} values, dtype={self._np_dtype()})"


class Datetime(_TimeArray):
    special_objType = "Datetime"

    def _np_dtype(self) -> str:
        return "datetime64[ns]"

    def _scalar_callback(self, scalar: int) -> pdTimestamp:
        return pdTimestamp(int(scalar), unit="ns")

    def __sub__(self, other):
        if isinstance(other, Datetime):
            return Timedelta(self.values._binop(other.values, "-"))
        if isinstance(other, Timedelta):
            return Datetime(self.values._binop(other.values, "-"))
        if isinstance(other, (pd.Timestamp, np.datetime64, _dt.datetime)):
            ns = normalize_to_ns(other)
            return Timedelta(self.values._binop(ns, "-"))
        if isinstance(other, (pd.Timedelta, np.timedelta64, _dt.timedelta)):
            ns = normalize_td_to_ns(other)
            return Datetime(self.values._binop(ns, "-"))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Timedelta):
            return Datetime(self.values._binop(other.values, "+"))
        if isinstance(other, (pd.Timedelta, np.timedelta64, _dt.timedelta)):
            ns = normalize_td_to_ns(other)
            return Datetime(self.values._binop(ns, "+"))
        return NotImplemented


class Timedelta(_TimeArray):
    special_objType = "Timedelta"

    def _np_dtype(self) -> str:
        return "timedelta64[ns]"

    def _scalar_callback(self, scalar: int) -> pdTimedelta:
        return pdTimedelta(int(scalar), unit="ns")

    def __add__(self, other):
        if isinstance(other, (Timedelta, pd.Timedelta, np.timedelta64, _dt.timedelta)):
            ns = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return Timedelta(self.values._binop(ns, "+"))
        if isinstance(other, (Datetime, pd.Timestamp, np.datetime64, _dt.datetime)):
            ns = other.values if isinstance(other, Datetime) else normalize_to_ns(other)
            return Datetime(self.values._binop(ns, "+"))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (Timedelta, pd.Timedelta, np.timedelta64, _dt.timedelta)):
            ns = other.values if isinstance(other, Timedelta) else normalize_td_to_ns(other)
            return Timedelta(self.values._binop(ns, "-"))
        return NotImplemented
