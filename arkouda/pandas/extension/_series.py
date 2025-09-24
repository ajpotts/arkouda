import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from arkouda.numpy.dtypes import bool_ as ak_bool_
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.pdarraycreation import pdarray
from arkouda.numpy.pdarraycreation import zeros as ak_zeros

# --- Minimal Arkouda Int64 dtype/array (adapt to your project’s versions) -----


class ArkoudaInt64Dtype(ExtensionDtype):
    name = "akint64"
    type = int
    kind = "i"

    @property
    def na_value(self):
        return pd.NA

    @classmethod
    def construct_array_type(cls):
        return ArkoudaInt64Array


class ArkoudaInt64Array(ExtensionArray):
    def __init__(self, data: pdarray):
        self._data = data

    @property
    def dtype(self):
        return ArkoudaInt64Dtype()

    def __len__(self):
        return self._data.size

    def __getitem__(self, i):
        return self._data[i]

    def isna(self):
        return ak_zeros(len(self), dtype=ak_bool_)  # tweak if you support NA

    def take(self, idx, allow_fill=False, fill_value=None):
        # You may want to implement allow_fill semantics fully
        return type(self)(self._data[idx])

    def copy(self):
        return type(self)(self._data)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(ak_array(scalars))


# Optional: a quick coercion helper (extend for Strings, Categorical, etc.)
def _to_ak_extarray(obj):
    if isinstance(obj, ArkoudaInt64Array):
        return obj
    if isinstance(obj, pdarray):
        return ArkoudaInt64Array(obj)
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
        if isinstance(values, ArkoudaInt64Array):
            return values._data
        raise TypeError("Series is not backed by an Arkouda ExtensionArray")
