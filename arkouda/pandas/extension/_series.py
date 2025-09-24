import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from arkouda.numpy.dtypes import bool_ as ak_bool_
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraycreation import pdarray
from arkouda.numpy.pdarraycreation import zeros as ak_zeros
from arkouda.pandas.extension import ArkoudaArray


# Optional: a quick coercion helper (extend for Strings, Categorical, etc.)
def _to_ak_extarray(obj):
    if isinstance(obj, ArkoudaArray):
        return obj
    if isinstance(obj, pdarray):
        return ArkoudaArray(obj)
    # Add branches for ak.Strings, ak.Categorical -> ArkoudaStringArray, ArkoudaCategoricalArray
    raise TypeError(f"Unsupported Arkouda type: {type(obj)}")


# ----------------------------- Option A: No subclass ---------------------------
# You can already do:
# s = pd.Series(ArkoudaInt64Array(ak.array([10, 20, 30])), name="x")
# s.dtype is ArkoudaInt64Dtype(), and pandas will delegate ops to your ExtensionArray.


# ----------------------------- Option B: Subclass ------------------------------
class ArkoudaSeries(pd.Series):
    """
    A thin wrapper around pandas.Series that:
      * accepts Arkouda arrays directly
      * preserves subclass through pandas ops
      * exposes helpers to round-trip Arkouda data
    Internals still use your ExtensionArray — no NumPy copy required.
    """

    # Make sure pandas returns ArkoudaSeries after ops
    @property
    def _constructor(self):
        return ArkoudaSeries

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        # Convert Arkouda input to your ExtensionArray and carry its dtype
        if isinstance(data, (pdarray, ArkoudaInt64Array)):
            data = _to_ak_extarray(data)
            dtype = data.dtype
        super().__init__(data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)

    @property
    def _constructor_expanddim(self):
        # If you later create an ArkoudaDataFrame subclass, return it here.
        return pd.DataFrame

    # -------- ergonomics --------
    @classmethod
    def from_ak(cls, a, name=None):
        """Create from an Arkouda array (e.g., ak.pdarray or ak.Strings once you add that)."""
        return cls(_to_ak_extarray(a), name=name)

    def to_ak(self):
        """Return the underlying Arkouda array if present, else raise."""
        values = self._values  # pandas’ EA stays here; do NOT rely on .values (may convert)
        if isinstance(values, ArkoudaArray):
            return values._data
        raise TypeError("Series is not backed by an Arkouda ExtensionArray")
