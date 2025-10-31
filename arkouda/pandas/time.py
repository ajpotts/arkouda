from pandas import date_range as pd_date_range
from pandas import timedelta_range as pd_timedelta_range

from arkouda.numpy import Datetime, Timedelta


def timedelta_range(start=None, end=None, periods=None, freq=None, name=None, closed=None, **kwargs):
    """
    Return a fixed frequency TimedeltaIndex, with day as the default
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

def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize=False,
    name=None,
    inclusive="both",
    **kwargs,
):
    """
    Create a fixed frequency Datetime range. Alias for
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
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries. Whether to set each bound as closed or open.
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
    return Datetime(
        pd_date_range(
            start,
            end,
            periods,
            freq,
            tz,
            normalize,
            name,
            inclusive=inclusive,
            **kwargs,
        )
    )

