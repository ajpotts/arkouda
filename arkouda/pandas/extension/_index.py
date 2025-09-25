import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.core.indexes.base import Index


from ._arkouda_array import ArkoudaArray
import pandas as pd
from pandas.api.extensions import register_index_accessor
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.strings import Strings
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.pandas.categorical import Categorical
import numpy as np

class ArkoudaIndex(pd.Index):
    _typ = "arkoudaindex"
    _data: ArkoudaArray

    def __new__(cls, data, dtype=None, copy=False, name=None):
        if not isinstance(data, ArkoudaArray):
            data = ArkoudaArray(ak_array(data))
        return super().__new__(cls, data, dtype=dtype, copy=copy, name=name)

# --- helpers --------------------------------------------------------

def _ak_underlying(idx: pd.Index):
    arr = getattr(idx, "array", None) or idx._values
    return arr._data if hasattr(arr, "_data") else ak_array(idx.to_numpy(object))

def _wrap_index_from_ak(ak_obj, *, name):
    # Detect Arkouda type and wrap in your EA via pd.array(..., dtype=...)
    if isinstance(ak_obj, pdarray):
        return pd.Index(pd.array(ak_obj, dtype="akint64"), name=name)
    if  isinstance(ak_obj, Strings):
        # requires your ArkoudaStringDtype to be registered
        return pd.Index(pd.array(ak_obj, dtype="akstring"), name=name)
    if  isinstance(ak_obj, Categorical):
        return pd.Index(pd.array(ak_obj, dtype="akcategory"), name=name)
    # Fallback (not EA-backed): materialize
    vals = ak_obj.to_ndarray() if hasattr(ak_obj, "to_ndarray") else np.asarray(ak_obj)
    return pd.Index(vals, name=name)

@register_index_accessor("ak")
class _AkIndexAccessor:
    def __init__(self, idx: pd.Index):
        self._idx = idx

    # Example: fast union for ints/strings using Arkouda
    def union(self, other: pd.Index, *, sort=None) -> pd.Index:
        a = self._as_ak(self._idx)
        b = self._as_ak(other)
        from arkouda import unique as ak_unique, concatenate as ak_concatenate
        out = ak_unique(ak_concatenate([a, b]))  # adjust if you need stable order
        return _wrap_index_from_ak(out, name=self._idx.name)

    # Example: fast intersection
    def intersection(self, other: pd.Index, *, sort=False) -> pd.Index:
        a = self._as_ak(self._idx); b = self._as_ak(other)
        from arkouda import intersect1d as ak_intersect1d, in1d as ak_in1d, unique as ak_unique
        try:
            inter = ak_intersect1d(a, b)
        except AttributeError:
            inter = a[ak_in1d(a, b)]
            inter = ak_unique(inter)
        return _wrap_index_from_ak(inter, name=self._idx.name)

    # Sketch: scalable get_indexer (exact)
    def get_indexer(self, target: pd.Index) -> pd.Index:
        src = self._as_ak(self._idx)
        tgt = self._as_ak(target)
        # Fast path if you have ak.unique(..., return_inverse=True)
        # u_src, inv_src = ak.unique(src, return_inverse=True)
        # pos = self._first_positions(inv_src)  # implement via GroupBy/segments
        # loc = ak.lookup(u_src, pos, tgt, default=-1)  # or equivalent
        # return ak.to_ndarray(loc)
        # For now, fall back to pandas for correctness:
        return self._idx.get_indexer(target)

    @staticmethod
    def _as_ak(idx: pd.Index):
        # If EA-backed, prefer .array (pandas 2.x public) to avoid .values materialization
        arr = getattr(idx, "array", None) or idx._values
        if hasattr(arr, "_data"):                 # your Arkouda EA convention
            return arr._data                      # ak.pdarray / ak.Strings / etc.
        # Else materialize (last resort)
        import numpy as np
        return ak_array(np.asarray(idx, dtype=object))
