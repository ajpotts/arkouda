from typing import List, Optional, Union

import pandas as pd
from pandas import DataFrame as pd_DataFrame
from pandas.api.extensions import register_dataframe_accessor
from typeguard import typechecked

from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.strings import Strings
from arkouda.pandas.categorical import Categorical
from arkouda.pandas.dataframe import DataFrame as ak_DataFrame
from arkouda.pandas.extension._arkouda_base_array import ArkoudaBaseArray

from ._arkouda_array import ArkoudaArray
from ._dtypes import _ArkoudaBaseDtype


def _looks_like_ak_col(obj) -> bool:
    return isinstance(obj, (pdarray, Strings, Categorical))


def _extract_ak_from_ea(ea):
    # Try common attributes first
    if hasattr(ea, "_data"):
        col = getattr(ea, "_data", None)
        if _looks_like_ak_col(col):
            return col
    # Then try a method hook
    for meth in ("to_ak", "__arkouda__", "ak_view"):
        fn = getattr(ea, meth, None)
        if callable(fn):
            col = fn()
            if _looks_like_ak_col(col):
                return col
    raise TypeError("Arkouda EA does not expose an Arkouda column via a known attribute/method.")


def _is_arkouda_series(s: pd.Series) -> bool:
    # dtype check first; fallback to EA instance name if helpful
    if isinstance(getattr(s, "dtype", None), _ArkoudaBaseDtype):
        return True
    return isinstance(getattr(s, "array", None), ArkoudaArray)


def _series_to_akcol_no_copy(s: pd.Series):
    if not _is_arkouda_series(s):
        raise TypeError(
            f"Column '{s.name}' is not Arkouda-backed (dtype={s.dtype!r}). "
            "Wrap columns with ArkoudaArray before calling df.ak.merge."
        )
    return _extract_ak_from_ea(s.array)


def _series_to_akcol_no_copy(s: pd.Series):
    """
    Extract the underlying Arkouda column from an Arkouda-backed pandas.Series
    without converting to NumPy/Python.
    """
    if not _is_arkouda_series(s):
        raise TypeError(f"Column '{s.name}' is not Arkouda-backed (got dtype={s.dtype!r}).")
    ea = s.array
    if hasattr(ea, "_data"):
        return ea._data
    raise TypeError(f"Cannot find underlying Arkouda array for column '{s.name}'.")


def _akcol_to_series(name: str, akcol) -> pd.Series:
    """
    Wrap an Arkouda column back into a pandas.Series using ArkoudaArray
    without converting to NumPy/Python.
    """
    # ArkoudaArray should provide a zero-copy-ish wrapper constructor from akcol
    ea = ArkoudaArray._from_sequence(akcol)
    return pd.Series(ea, name=name)


def _df_to_akdf_no_copy(df: pd.DataFrame) -> ak_DataFrame:
    """
    Convert a *pandas* DataFrame (all Arkouda-backed columns) to ak.DataFrame
    without NumPy/Python conversion.
    """
    cols = {}
    for name in df.columns:
        s = df[name]
        cols[name] = _series_to_akcol_no_copy(s)
    return ak_DataFrame(cols)


def _akdf_to_pandas_no_copy(akdf: "ak.DataFrame") -> pd.DataFrame:
    """
    Convert an ak.DataFrame back to pandas.DataFrame with Arkouda ExtensionArrays
    (no NumPy/Python conversion).
    """
    cols = {}
    for name in akdf.columns:
        cols[name] = _akcol_to_series(name, akdf[name])
    return pd.DataFrame(cols)


@register_dataframe_accessor("ak")
class ArkoudaDataFrameAccessor:
    """
    Bare-bones Arkouda DataFrame accessor.

    Allows `df.ak` access to Arkouda-backed operations.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def to_ak(self):
        """
        Convert the pandas DataFrame to an Arkouda DataFrame.
        """
        cols = {}
        for name, col in self._obj.items():
            cols[name] = ArkoudaArray(ak_array(col.values))
        return pd_DataFrame(cols)

    def from_ak(self, akdf):
        """
        Convert an Arkouda DataFrame back to pandas.
        """
        import pandas as pd

        cols = {name: akdf[name].to_ndarray() for name in akdf.columns}
        return pd.DataFrame(cols)

    def info(self):
        """
        Simple Arkouda-like summary of the underlying pandas DataFrame.
        """
        print(f"Arkouda Accessor for DataFrame with {len(self._obj)} rows")
        print("Columns:", list(self._obj.columns))

    # Optional: simple sanity check
    def _assert_all_arkouda(self):
        non_ak = [c for c in self._obj.columns if not _is_arkouda_series(self._obj[c])]
        if non_ak:
            raise TypeError(
                "All columns must be Arkouda-backed ExtensionArrays for df.ak.merge.\n"
                f"Non-Arkouda columns: {non_ak}"
            )

    def _assert_all_arkouda(self, df: pd.DataFrame, side: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"{side} must be a pandas.DataFrame; got {type(df).__name__}. "
                "If you already have ak.DataFrame, call left_ak.merge(right_ak, ...)."
            )
        bad = [c for c in df.columns if not _is_arkouda_series(df[c])]
        if bad:
            raise TypeError(
                f"All columns in the {side} DataFrame must be Arkouda ExtensionArrays. "
                f"Non-Arkouda columns: {bad}"
            )

    @typechecked
    def merge(
        self,
        right: pd.DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
        left_suffix: str = "_x",
        right_suffix: str = "_y",
        convert_ints: bool = True,
        sort: bool = True,
    ) -> pd.DataFrame:
        r"""
        Merge two pandas DataFrames (both Arkouda-backed) using Arkouda's join,
        returning a pandas.DataFrame with Arkouda ExtensionArrays.

        No NumPy/Python materialization is performed.

        Parameters
        ----------
        right : pandas.DataFrame
            Right table; must be Arkouda-backed (per-column).
        on : str or list of str, optional
            Join key(s). If None, uses intersection of columns.
        how : {"inner","left","right","outer"}, default "inner"
            Join type.
        left_suffix : str, default "_x"
            Suffix for overlapping left columns (when needed).
        right_suffix : str, default "_y"
            Suffix for overlapping right columns (when needed).
        convert_ints : bool, default True
            Match pandas behavior for missing integers (if applicable in backend).
        sort : bool, default True
            Sort result by `on`.

        Returns
        -------
        pandas.DataFrame
            Result with Arkouda ExtensionArrays.

        Notes
        -----
        - Multiple-key joins may be limited to integer keys depending on backend.
        - This path requires *all* participating columns to be Arkouda-backed.
        """
        if not isinstance(right, pd.DataFrame):
            raise TypeError("`right` must be a pandas.DataFrame.")

        # Ensure both sides are Arkouda-backed
        self._assert_all_arkouda()
        non_ak_right = [c for c in right.columns if not _is_arkouda_series(right[c])]
        if non_ak_right:
            raise TypeError(
                "All columns in `right` must be Arkouda-backed ExtensionArrays.\n"
                f"Non-Arkouda columns in right: {non_ak_right}"
            )

        # Lift to ak.DataFrame without NumPy
        left_ak = _df_to_akdf_no_copy(self._obj)
        right_ak = _df_to_akdf_no_copy(right)

        # Delegate to Arkouda merge
        out_ak = left_ak.merge(
            right_ak,
            on=on,
            how=how,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
            convert_ints=convert_ints,
            sort=sort,
        )

        # Wrap back into pandas with Arkouda EAs (no NumPy)
        return _akdf_to_pandas_no_copy(out_ak)

    @typechecked
    def merge(
        self,
        right: pd.DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
        left_suffix: str = "_x",
        right_suffix: str = "_y",
        convert_ints: bool = True,
        sort: bool = True,
    ) -> pd.DataFrame:
        r"""
        Merge two pandas DataFrames (both Arkouda-backed) using Arkouda's join,
        returning a pandas.DataFrame with Arkouda EAs. No NumPy materialization.

        Parameters match Arkouda/DataFrame.merge: on/left_on/right_on/how/suffixes/convert_ints/sort.
        """
        if not isinstance(right, pd.DataFrame):
            raise TypeError("`right` must be a pandas.DataFrame")

        # Ensure both sides are Arkouda-backed
        self._assert_all_arkouda(self._obj, "left")
        self._assert_all_arkouda(right, "right")

        # Lift to ak.DataFrame (zero-copy-ish)
        left_ak = _df_to_akdf_no_copy(self._obj)
        right_ak = _df_to_akdf_no_copy(right)

        # Normalize join keys similar to Arkouda/pandas merge semantics
        col_intersect = list(set(left_ak.columns) & set(right_ak.columns))
        if on is None and left_on is None and right_on is None:
            _on = col_intersect
            left_on_, right_on_ = _on, _on
        else:
            _on = on
            if (left_on is None) != (right_on is None):
                raise ValueError("If one of left_on/right_on is set, the other must also be set.")
            left_on_ = left_on if left_on is not None else _on
            right_on_ = right_on if right_on is not None else _on

        from arkouda.pandas.dataframe import merge

        # Delegate to the ak.DataFrame instance method (NOT a top-level ak_merge)
        out_ak = merge(
            left_ak,
            right_ak,
            on=_on,
            left_on=left_on_,
            right_on=right_on_,
            how=how,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
            convert_ints=convert_ints,
            sort=sort,
        )

        # Wrap back into pandas with Arkouda EAs
        return _akdf_to_pandas_no_copy(out_ak)
