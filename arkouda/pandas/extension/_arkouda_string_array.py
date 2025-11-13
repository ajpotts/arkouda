import numpy as np
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.strings import Strings

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import ArkoudaStringDtype


__all__ = ["ArkoudaStringArray"]


class ArkoudaStringArray(ArkoudaExtensionArray, ExtensionArray):
    default_fill_value = ""

    def __init__(self, data):
        """
        Initialize an Arkouda-backed ExtensionArray for string data.

        This constructor ensures that the underlying data is an Arkouda
        `Strings` object, which represents a distributed string array
        stored on the Arkouda server. It accepts data from compatible
        Python or NumPy objects and converts them if needed.

        Parameters
        ----------
        data : arkouda.Strings, numpy.ndarray, list, tuple, or ArkoudaStringArray
            Input data to wrap or convert.

            - If `data` is an `arkouda.Strings` object, it is used directly.
            - If `data` is a NumPy array, Python list, or tuple of strings,
              it is converted to an Arkouda `Strings` object via `ak.array()`.
            - If `data` is another `ArkoudaStringArray`, its backing `Strings`
              array is reused directly.

        Raises
        ------
        TypeError
            If the input cannot be converted to an Arkouda `Strings` object.

        Notes
        -----
        - Arkouda `Strings` arrays are immutable, so setting `copy=True` is not
          required for correctness but may be useful to ensure isolation from
          shared references in derived operations.
        - For one-dimensional string arrays, this class provides pandas
          ExtensionArray compatibility (e.g., `.astype`, `.isna`, `.copy`,
          `.equals`).

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaStringArray
        >>> s = ak.array(["apple", "banana", "cherry"])
        >>> ArkoudaStringArray(s)
        ArkoudaStringArray(['apple', 'banana', 'cherry'])

        >>> import numpy as np
        >>> ArkoudaStringArray(np.array(["red", "green", "blue"]))
        ArkoudaStringArray(['red', 'green', 'blue'])

        >>> ArkoudaStringArray(["x", "y", "z"])
        ArkoudaStringArray(['x', 'y', 'z'])
        """
        from arkouda.numpy.pdarraycreation import array as ak_array

        # Reuse backing data if another ArkoudaStringArray
        if isinstance(data, ArkoudaStringArray):
            self._data = data._data
            return

        # Convert numpy arrays, lists, or tuples of strings
        if isinstance(data, (np.ndarray, list, tuple)):
            data = ak_array(data, dtype="str_")

        # Validate type
        if not isinstance(data, Strings):
            raise TypeError(f"Expected arkouda.Strings, got {type(data).__name__}")

        self._data = data

    @property
    def dtype(self):
        return ArkoudaStringDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(ak_array(scalars))

    def __getitem__(self, key):
        result = self._data[key]
        if np.isscalar(key):
            if hasattr(result, "to_ndarray"):
                return result.to_ndarray()[()]
            else:
                return result
        return ArkoudaStringArray(result)

    def astype(self, dtype, copy: bool = False):
        if dtype in (object, np.object_, "object", np.dtype("O")):
            return self.to_ndarray().astype(object, copy=copy)
        # Let pandas do the rest locally
        return self.to_ndarray().astype(dtype, copy=copy)

    def isna(self):
        from arkouda.numpy.pdarraycreation import zeros

        return zeros(self._data.size, dtype="bool")

    def copy(self):
        return ArkoudaStringArray(self._data[:])

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaStringArray) else other)

    def __repr__(self):
        return f"ArkoudaStringArray({self._data})"
