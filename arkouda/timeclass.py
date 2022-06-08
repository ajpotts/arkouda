import datetime
from typing import Union

import numpy as np  # type: ignore
from pandas import Series  # type: ignore
from pandas import Timedelta as pdTimedelta  # type: ignore
from pandas import Timestamp  # type: ignore
from pandas import date_range as pd_date_range  # type: ignore
from pandas import timedelta_range as pd_timedelta_range  # type: ignore
from pandas import to_datetime, to_timedelta  # type: ignore

from arkouda.dtypes import int64, intTypes, isSupportedInt
from arkouda.numeric import abs as akabs
from arkouda.numeric import cast
from arkouda.pdarrayclass import pdarray
from arkouda.pdarraycreation import from_series

_BASE_UNIT = "ns"

_unit2normunit = {
    "weeks": "w",
    "days": "d",
    "hours": "h",
    "hrs": "h",
    "minutes": "m",
    "t": "m",
    "milliseconds": "ms",
    "l": "ms",
    "microseconds": "us",
    "u": "us",
    "nanoseconds": "ns",
    "n": "ns",
}

_unit2factor = {
    "w": 7 * 24 * 60 * 60 * 10**9,
    "d": 24 * 60 * 60 * 10**9,
    "h": 60 * 60 * 10**9,
    "m": 60 * 10**9,
    "s": 10**9,
    "ms": 10**6,
    "us": 10**3,
    "ns": 1,
}


def _get_factor(unit: str) -> int:
    unit = unit.lower()
    if unit in _unit2factor:
        return _unit2factor[unit]
    else:
        for key, normunit in _unit2normunit.items():
            if key.startswith(unit):
                return _unit2factor[normunit]
        raise ValueError(
            f"Argument must be one of {set(_unit2factor.keys()) | set(_unit2normunit.keys())}"
        )


def _identity(x, **kwargs):
    return x


class _Timescalar:
    def __init__(self, scalar):
        if isinstance(scalar, np.datetime64) or isinstance(scalar, datetime.datetime):
            scalar = to_datetime(scalar).to_numpy()
        elif isinstance(scalar, np.timedelta64) or isinstance(scalar, datetime.timedelta):
            scalar = to_timedelta(scalar).to_numpy()
        self.unit = np.datetime_data(scalar.dtype)[0]
        self._factor = _get_factor(self.unit)
        # int64 in nanoseconds
        self.value = self._factor * scalar.astype("int64")


class _AbstractBaseTime(pdarray):
    """Base class for Datetime and Timedelta; not user-facing. Arkouda handles
    time similar to Pandas (albeit with less functionality), in that all absolute
    and relative times are represented in nanoseconds as int64 behind the scenes.
    Datetime and Timedelta can be constructed from Arkouda, NumPy, or Pandas arrays;
    in each case, the input values are normalized to nanoseconds on initialization,
    so that all resulting operations are transparent.
    """

    def __init__(self, array, unit: str = _BASE_UNIT):  # type: ignore

        if isinstance(array, Datetime) or isinstance(array, Timedelta):
            self.unit: str = array.unit
            self._factor: int = array._factor
            # Make a copy to avoid unknown symbol errors
            self.values: pdarray = cast(array.values, int64)
        # Convert the input to int64 pdarray of nanoseconds
        elif isinstance(array, pdarray):
            if array.dtype not in intTypes:
                raise TypeError(f"{self.__class__.__name__} array must have int64 dtype")
            # Already int64 pdarray, just scale
            self.unit = unit
            self._factor = _get_factor(self.unit)
            # This makes a copy of the input array, to leave input unchanged
            self.values = cast(self._factor * array, int64)  # Mimics a datetime64[ns] array
        elif hasattr(array, "dtype"):
            # Handles all pandas and numpy datetime/timedelta arrays
            if array.dtype.kind not in ("M", "m"):
                # M = datetime64, m = timedelta64
                raise TypeError(f"Invalid dtype: {array.dtype.name}")
            if isinstance(array, Series):
                # Pandas Datetime and Timedelta
                # Get units of underlying numpy datetime64 array
                self.unit = np.datetime_data(array.values.dtype)[0]
                self._factor = _get_factor(self.unit)
                # Create pdarray
                self.values = from_series(array)
                # Scale if necessary
                # This is futureproofing; it will not be used unless pandas
                # changes its Datetime implementation
                if self._factor != 1:
                    # Scale inplace because we already created a copy
                    self.values *= self._factor
            elif isinstance(array, np.ndarray):
                # Numpy datetime64 and timedelta64
                # Force through pandas.Series
                self.__init__(to_datetime(array).to_series())  # type: ignore
            elif hasattr(array, "to_series"):
                # Pandas DatetimeIndex
                # Force through pandas.Series
                self.__init__(array.to_series())  # type: ignore
        else:
            raise TypeError(f"Unsupported type: {type(array)}")
        # Now that self.values is correct, init self with same metadata
        super().__init__(
            self.values.name,
            self.values.dtype,
            self.values.size,
            self.values.ndim,
            self.values.shape,
            self.values.itemsize,
        )
        # Deprecated
        self._data = self.values

    @classmethod
    def _get_callback(cls, other, op):
        # Will be overridden by all children
        return _identity

    def floor(self, freq):
        """Round times down to the nearest integer of a given frequency.

        Parameters
        ----------
        freq : str {'d', 'm', 'h', 's', 'ms', 'us', 'ns'}
            Frequency to round to

        Returns
        -------
        self.__class__
            Values rounded down to nearest frequency
        """
        f = _get_factor(freq)
        return self.__class__(self.values // f, unit=freq)

    def ceil(self, freq):
        """Round times up to the nearest integer of a given frequency.

        Parameters
        ----------
        freq : str {'d', 'm', 'h', 's', 'ms', 'us', 'ns'}
            Frequency to round to

        Returns
        -------
        self.__class__
            Values rounded up to nearest frequency
        """
        f = _get_factor(freq)
        return self.__class__((self.values + (f - 1)) // f, unit=freq)

    def round(self, freq):
        """Round times to the nearest integer of a given frequency. Midpoint
        values will be rounded to nearest even integer.

        Parameters
        ----------
        freq : str {'d', 'm', 'h', 's', 'ms', 'us', 'ns'}
            Frequency to round to

        Returns
        -------
        self.__class__
            Values rounded to nearest frequency
        """
        f = _get_factor(freq)
        offset = self.values + ((f + 1) // 2)
        rounded = offset // f
        # Halfway values are supposed to round to the nearest even integer
        # Need to figure out which ones ended up odd and fix them
        decrement = ((offset % f) == 0) & ((rounded % 2) == 1)
        rounded[decrement] = rounded[decrement] - 1
        return self.__class__(rounded, unit=freq)

    def to_ndarray(self):
        __doc__ = super().to_ndarray.__doc__  # noqa
        return np.array(
            self.values.to_ndarray(), dtype="{}64[ns]".format(self.__class__.__name__.lower())
        )

    def __str__(self):
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            vals = [f"'{self[i]}'" for i in range(self.size)]
        else:
            vals = [f"'{self[i]}'" for i in range(3)]
            vals.append("... ")
            vals.extend([f"'{self[i]}'" for i in range(self.size - 3, self.size)])
        spaces = " " * (len(self.__class__.__name__) + 1)
        return "{}([{}],\n{}dtype='{}64[ns]')".format(
            self.__class__.__name__,
            ",\n{} ".format(spaces).join(vals),
            spaces,
            self.__class__.__name__.lower(),
        )

    def __repr__(self) -> str:
        return self.__str__()

    def _binop(self, other, op):
        # Need to do 2 things:
        #  1) Determine return type, based on other's class
        #  2) Get other's int64 data to combine with self's data
        if isinstance(other, Datetime) or self._is_datetime_scalar(other):
            if op not in self.supported_with_datetime:
                raise TypeError(f"{op} not supported between {self.__class__.__name__} and Datetime")
            otherclass = "Datetime"
            if self._is_datetime_scalar(other):
                otherdata = _Timescalar(other).value
            else:
                otherdata = other.values
        elif isinstance(other, Timedelta) or self._is_timedelta_scalar(other):
            if op not in self.supported_with_timedelta:
                raise TypeError(f"{op} not supported between {self.__class__.__name__} and Timedelta")
            otherclass = "Timedelta"
            if self._is_timedelta_scalar(other):
                otherdata = _Timescalar(other).value
            else:
                otherdata = other.values
        elif (isinstance(other, pdarray) and other.dtype in intTypes) or isSupportedInt(other):
            if op not in self.supported_with_pdarray:
                raise TypeError(f"{op} not supported between {self.__class__.__name__} and integer")
            otherclass = "pdarray"
            otherdata = other
        else:
            return NotImplemented
        # Determines return type (Datetime, Timedelta, or pdarray)
        callback = self._get_callback(otherclass, op)
        # Actual operation evaluates on the underlying int64 data
        return callback(self.values._binop(otherdata, op))

    def _r_binop(self, other, op):
        # Need to do 2 things:
        #  1) Determine return type, based on other's class
        #  2) Get other's int64 data to combine with self's data

        # First case is pdarray <op> self
        if isinstance(other, pdarray) and other.dtype in intTypes:
            if op not in self.supported_with_r_pdarray:
                raise TypeError(f"{op} not supported between int64 and {self.__class__.__name__}")
            callback = self._get_callback("pdarray", op)
            # Need to use other._binop because self.values._r_binop can only handle scalars
            return callback(other._binop(self.values, op))
        # All other cases are scalars, so can use self.values._r_binop
        elif self._is_datetime_scalar(other):
            if op not in self.supported_with_r_datetime:
                raise TypeError(
                    f"{op} not supported between scalar datetime and {self.__class__.__name__}"
                )
            otherclass = "Datetime"
            otherdata = _Timescalar(other).value
        elif self._is_timedelta_scalar(other):
            if op not in self.supported_with_r_timedelta:
                raise TypeError(
                    f"{op} not supported between scalar timedelta and {self.__class__.__name__}"
                )
            otherclass = "Timedelta"
            otherdata = _Timescalar(other).value
        elif isSupportedInt(other):
            if op not in self.supported_with_r_pdarray:
                raise TypeError(f"{op} not supported between int64 and {self.__class__.__name__}")
            otherclass = "pdarray"
            otherdata = other
        else:
            # If here, type is not handled
            return NotImplemented
        callback = self._get_callback(otherclass, op)
        return callback(self.values._r_binop(otherdata, op))

    def opeq(self, other, op):
        if isinstance(other, Timedelta) or self._is_timedelta_scalar(other):
            if op not in self.supported_opeq:
                raise TypeError(f"{self.__class__.__name__} {op} Timedelta not supported")
            if self._is_timedelta_scalar(other):
                otherdata = _Timescalar(other).value
            else:
                otherdata = other.values
            self.values.opeq(otherdata, op)
        elif isinstance(other, Datetime) or self._is_datetime_scalar(other):
            raise TypeError(f"{self.__class__.__name__} {op} datetime not supported")
        else:
            return NotImplemented

    @staticmethod
    def _is_datetime_scalar(scalar):
        return (
            isinstance(scalar, Timestamp)
            or (isinstance(scalar, np.datetime64) and np.isscalar(scalar))
            or isinstance(scalar, datetime.datetime)
        )

    @staticmethod
    def _is_timedelta_scalar(scalar):
        return (
            isinstance(scalar, pdTimedelta)
            or (isinstance(scalar, np.timedelta64) and np.isscalar(scalar))
            or isinstance(scalar, datetime.timedelta)
        )

    def _scalar_callback(self, key):
        # Will be overridden in all children
        return key

    def __getitem__(self, key):
        if isSupportedInt(key):
            # Single integer index will return a pandas scalar
            return self._scalar_callback(self.values[key])
        else:
            # Slice or array index should return same class
            return self.__class__(self.values[key])

    def __setitem__(self, key, value):
        # RHS can only be vector or scalar of same class
        if isinstance(value, self.__class__):
            # Value.values is already in nanoseconds, so self.values
            # can be set directly
            self.values[key] = value.values
        elif self._is_supported_scalar(value):
            # _Timescalar takes care of normalization to nanoseconds
            normval = _Timescalar(value)
            self.values[key] = normval.value
        else:
            return NotImplemented

    def min(self):
        __doc__ = super().min.__doc__  # noqa
        # Return type is pandas scalar
        return self._scalar_callback(self.values.min())

    def max(self):
        __doc__ = super().max.__doc__  # noqa
        # Return type is pandas scalar
        return self._scalar_callback(self.values.max())

    def mink(self, k):
        __doc__ = super().mink.__doc__  # noqa
        # Return type is same class
        return self.__class__(self.values.mink(k))

    def maxk(self, k):
        __doc__ = super().maxk.__doc__  # noqa
        # Return type is same class
        return self.__class__(self.values.maxk(k))


class Datetime(_AbstractBaseTime):
    """Represents a date and/or time.

    Datetime is the Arkouda analog to pandas DatetimeIndex and
    other timeseries data types.

    Parameters
    ----------
    array : int64 pdarray, pd.DatetimeIndex, pd.Series, or np.datetime64 array
    uint : str, default 'ns'
        For int64 pdarray, denotes the unit of the input. Ignored for pandas
        and numpy arrays, which carry their own unit. Not case-sensitive;
        prefixes of full names (like 'sec') are accepted.

        Possible values:

        * 'weeks' or 'w'
        * 'days' or 'd'
        * 'hours' or 'h'
        * 'minutes', 'm', or 't'
        * 'seconds' or 's'
        * 'milliseconds', 'ms', or 'l'
        * 'microseconds', 'us', or 'u'
        * 'nanoseconds', 'ns', or 'n'

        Unlike in pandas, units cannot be combined or mixed with integers

    Notes
    -----
    The ``.values`` attribute is always in nanoseconds with int64 dtype.
    """

    supported_with_datetime = frozenset(("==", "!=", "<", "<=", ">", ">=", "-"))
    supported_with_r_datetime = frozenset(("==", "!=", "<", "<=", ">", ">=", "-"))
    supported_with_timedelta = frozenset(("+", "-", "/", "//", "%"))
    supported_with_r_timedelta = frozenset(("+"))
    supported_opeq = frozenset(("+=", "-="))
    supported_with_pdarray = frozenset(())  # type: ignore
    supported_with_r_pdarray = frozenset(())  # type: ignore

    @classmethod
    def _get_callback(cls, otherclass, op):
        callbacks = {
            ("Datetime", "-"): Timedelta,  # Datetime - Datetime -> Timedelta
            ("Timedelta", "+"): cls,  # Datetime + Timedelta -> Datetime
            ("Timedelta", "-"): cls,  # Datetime - Timedelta -> Datetime
            ("Timedelta", "%"): Timedelta,
        }  # Datetime % Timedelta -> Timedelta
        # Every other supported op returns an int64 pdarray, so callback is identity
        return callbacks.get((otherclass, op), _identity)

    def _scalar_callback(self, scalar):
        # Formats a scalar return value as pandas Timestamp
        return Timestamp(int(scalar), unit=_BASE_UNIT)

    @staticmethod
    def _is_supported_scalar(self, scalar):
        # Tests whether scalar has compatible type with self's elements
        return self.is_datetime_scalar(scalar)

    def to_pandas(self):
        """Convert array to a pandas DatetimeIndex. Note: if the array size
        exceeds client.maxTransferBytes, a RuntimeError is raised.

        See Also
        --------
        to_ndarray
        """
        return to_datetime(self.to_ndarray())

    def sum(self):
        raise TypeError("Cannot sum datetime64 values")


class Timedelta(_AbstractBaseTime):
    """Represents a duration, the difference between two dates or times.

    Timedelta is the Arkouda equivalent of pandas.TimedeltaIndex.

    Parameters
    ----------
    array : int64 pdarray, pd.TimedeltaIndex, pd.Series, or np.timedelta64 array
    unit : str, default 'ns'
        For int64 pdarray, denotes the unit of the input. Ignored for pandas
        and numpy arrays, which carry their own unit. Not case-sensitive;
        prefixes of full names (like 'sec') are accepted.

        Possible values:

        * 'weeks' or 'w'
        * 'days' or 'd'
        * 'hours' or 'h'
        * 'minutes', 'm', or 't'
        * 'seconds' or 's'
        * 'milliseconds', 'ms', or 'l'
        * 'microseconds', 'us', or 'u'
        * 'nanoseconds', 'ns', or 'n'

        Unlike in pandas, units cannot be combined or mixed with integers

    Notes
    -----
    The ``.values`` attribute is always in nanoseconds with int64 dtype.
    """

    supported_with_datetime = frozenset(("+"))
    supported_with_r_datetime = frozenset(("+", "-", "/", "//", "%"))
    supported_with_timedelta = frozenset(("==", "!=", "<", "<=", ">", ">=", "+", "-", "/", "//", "%"))
    supported_with_r_timedelta = frozenset(("==", "!=", "<", "<=", ">", ">=", "+", "-", "/", "//", "%"))
    supported_opeq = frozenset(("+=", "-=", "%="))
    supported_with_pdarray = frozenset(("*", "//"))
    supported_with_r_pdarray = frozenset(("*"))

    @classmethod
    def _get_callback(cls, otherclass, op):
        callbacks = {
            ("Timedelta", "-"): cls,  # Timedelta - Timedelta -> Timedelta
            ("Timedelta", "+"): cls,  # Timedelta + Timedelta -> Timedelta
            ("Datetime", "+"): Datetime,  # Timedelta + Datetime -> Datetime
            ("Datetime", "-"): Datetime,  # Datetime - Timedelta -> Datetime
            ("Timedelta", "%"): cls,  # Timedelta % Timedelta -> Timedelta
            ("pdarray", "//"): cls,  # Timedelta // pdarray -> Timedelta
            ("pdarray", "*"): cls,
        }  # Timedelta * pdarray -> Timedelta
        # Every other supported op returns an int64 pdarray, so callback is identity
        return callbacks.get((otherclass, op), _identity)

    def _scalar_callback(self, scalar):
        # Formats a returned scalar as a pandas.Timedelta
        return pdTimedelta(int(scalar), unit=_BASE_UNIT)

    @staticmethod
    def _is_supported_scalar(self, scalar):
        return self.is_timedelta_scalar(scalar)

    def to_pandas(self):
        """Convert array to a pandas TimedeltaIndex. Note: if the array size
        exceeds client.maxTransferBytes, a RuntimeError is raised.

        See Also
        --------
        to_ndarray
        """
        return to_timedelta(self.to_ndarray())

    def std(self, ddof: Union[int, np.int64, np.uint64] = 0):
        """
        Returns the standard deviation as a pd.Timedelta object
        """
        return self._scalar_callback(self.values.std(ddof=ddof))

    def sum(self):
        # Sum as a pd.Timedelta
        return self._scalar_callback(self.values.sum())

    def abs(self):
        """Absolute value of time interval."""
        return self.__class__(cast(akabs(self.values), "int64"))


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize=False,
    name=None,
    closed=None,
    **kwargs,
):
    """Creates a fixed frequency Datetime range. Alias for
    ``ak.Datetime(pd.date_range(args))``. Subject to size limit
    imposed by client.maxTransferBytes.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5H'. See
        timeseries.offset_aliases for a list of
        frequency aliases.
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        'Asia/Hong_Kong'. By default, the resulting DatetimeIndex is
        timezone-naive.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    closed : {None, 'left', 'right'}, optional
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None, the default).
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    rng : DatetimeIndex

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    """
    return Datetime(pd_date_range(start, end, periods, freq, tz, normalize, name, closed, **kwargs))


def timedelta_range(start=None, end=None, periods=None, freq=None, name=None, closed=None, **kwargs):
    """Return a fixed frequency TimedeltaIndex, with day as the default
    frequency. Alias for ``ak.Timedelta(pd.timedelta_range(args))``.
    Subject to size limit imposed by client.maxTransferBytes.

    Parameters
    ----------
    start : str or timedelta-like, default None
        Left bound for generating timedeltas.
    end : str or timedelta-like, default None
        Right bound for generating timedeltas.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5H'.
    name : str, default None
        Name of the resulting TimedeltaIndex.
    closed : str, default None
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None).

    Returns
    -------
    rng : TimedeltaIndex

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``TimedeltaIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.
    """
    return Timedelta(pd_timedelta_range(start, end, periods, freq, name, closed, **kwargs))
