from collections.abc import Mapping
from typing import Any, Dict, Iterable

import pandas as pd

from arkouda.numpy.pdarraycreation import pdarray
from arkouda.pandas.extension import ArkoudaArray

from ._series import ArkoudaSeries

# ---- Your existing pieces (sketch) -------------------------------------------
# from ._series import ArkoudaSeries, _to_ak_extarray
# from ._arrays import ArkoudaInt64Array, ArkoudaStringArray, ArkoudaCategoricalArray  # if implemented


# Minimal version of _to_ak_extarray for ints; extend for Strings/Categorical.
def _to_ak_extarray(obj):
    from pandas.api.extensions import ExtensionArray

    # Import here to avoid circulars in your package layout

    if isinstance(obj, ArkoudaArray):
        return obj
    if isinstance(obj, pdarray):
        if ArkoudaArray is None:
            raise TypeError("ArkoudaInt64Array not available")
        return ArkoudaArray(obj)

    # TODO: add:
    # if isinstance(obj, ak.Strings): return ArkoudaStringArray(obj)
    # if isinstance(obj, ak.Categorical): return ArkoudaCategoricalArray(obj)
    raise TypeError(f"Unsupported Arkouda type for column: {type(obj)}")


class ArkoudaDataFrame(pd.DataFrame):
    """
    Thin wrapper around pandas.DataFrame that:
      * Accepts Arkouda arrays directly for columns.
      * Returns ArkoudaSeries on column selection.
      * Preserves subclass through common DataFrame ops.
    Internally, each Arkouda column is stored as a pandas ExtensionArray.
    """

    # Ensure pandas returns our subclass from DataFrame ops
    @property
    def _constructor(self):
        return ArkoudaDataFrame

    @property
    def _constructor_sliced(self):
        # Selecting a single column returns our ArkoudaSeries
        return ArkoudaSeries

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy: bool = False):
        # Coerce dict-like and sequence-like data into EAs where appropriate
        data = self._coerce_data_to_ea(data)
        # Index: allow the user to pass a normal pandas Index or your ArkoudaIndex
        # If you’ve built ArkoudaIndex/ArkoudaMultiIndex, you can just pass them through.
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)

    # ---------- Constructors / helpers -----------------------------------------
    @classmethod
    def from_ak(cls, data: Mapping[str, Any] | Iterable[Any], index=None):
        """
        Build from Arkouda inputs:
          - If 'data' is a dict: {col: ak_array or EA}
          - If 'data' is sequence of (name, ak_array) pairs, that works too.
        """
        coerced = cls._coerce_data_to_ea(data)
        return cls(coerced, index=index)

    def to_ak_cols(self) -> Dict[str, Any]:
        """
        Return a dict of {column_name: underlying Arkouda array} for columns that
        are backed by an Arkouda ExtensionArray. Non-Arkouda columns are skipped.
        """
        out: Dict[str, Any] = {}
        for name, ser in self.items():
            vals = ser._values  # EA stays here
            if hasattr(vals, "_data"):  # your Arkouda EA convention
                out[name] = vals._data
        return out

    # ---------- Internal utilities ---------------------------------------------
    @staticmethod
    def _coerce_data_to_ea(data):
        """
        Convert Arkouda inputs into pandas-friendly structures with ExtensionArrays.
        Supports:
          * dict-like {col: ak.pdarray or ArkoudaEA or list/ndarray}
          * sequence of (name, value) pairs
          * already-a-DataFrame / Series -> returned unchanged
        """
        if data is None:
            return None

        # Pass-through: already a pandas object
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data

        # dict-like
        if isinstance(data, Mapping):
            out = {}
            for k, v in data.items():
                out[k] = ArkoudaDataFrame._maybe_coerce_col(v)
            return out

        # list of pairs or list/tuple of column arrays
        if isinstance(data, (list, tuple)):
            # If it looks like [("col", values), ...]
            if len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
                return [(k, ArkoudaDataFrame._maybe_coerce_col(v)) for (k, v) in data]
            # Otherwise let pandas handle it (e.g., 2D ndarray-like)
            return [ArkoudaDataFrame._maybe_coerce_col(v) for v in data]

        # Fallback: leave as-is (pandas will try to interpret)
        return data

    @staticmethod
    def _maybe_coerce_col(obj):
        """
        If the object is an Arkouda array, wrap it in the appropriate Arkouda EA.
        Otherwise, return as-is (NumPy array, list, scalar, etc.).
        """
        try:
            # Try Arkouda -> EA conversion
            return _to_ak_extarray(obj)
        except TypeError:
            # Not an Arkouda type we handle; leave it alone.
            return obj
