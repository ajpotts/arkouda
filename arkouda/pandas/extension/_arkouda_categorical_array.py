from typing import TYPE_CHECKING, TypeVar

from numpy import integer
from pandas.api.extensions import ExtensionArray

import arkouda as ak

from arkouda.numpy.dtypes import ARKOUDA_SUPPORTED_DTYPES, bool_, int64
from arkouda.numpy.pdarraycreation import array as ak_array

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import ArkoudaCategoricalDtype


if TYPE_CHECKING:
    from arkouda.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")


__all__ = ["ArkoudaCategoricalArray"]


class ArkoudaCategoricalArray(ArkoudaExtensionArray, ExtensionArray):
    default_fill_value: str = ""

    def __init__(self, data):
        if not isinstance(data, ak.Categorical):
            raise TypeError("Expected arkouda Categorical")
        self._data = data

    def __getitem__(self, key):
        """
        Retrieve one or more categorical values.

        Parameters
        ----------
        key : int, list, numpy.ndarray, slice, tuple, or arkouda indexer

        Returns
        -------
        scalar or ArkoudaCategoricalArray
        """
        from arkouda.pandas.categorical import Categorical

        #   If key is empty, return empty result
        if not key:
            return ArkoudaCategoricalArray(Categorical(ak_array([], dtype="str_")))

        if not isinstance(key, ARKOUDA_SUPPORTED_DTYPES):
            key = ak_array(key)

        # Delegate to underlying arkouda.Categorical
        result = self._data[key]
        if isinstance(result, ARKOUDA_SUPPORTED_DTYPES):
            return result
        if not isinstance(result, Categorical):
            result = Categorical(ak_array(result))

        # wrap it in an ArkoudaCategoricalArray.
        return ArkoudaCategoricalArray(result)

    def __setitem__(self, key, value) -> None:
        """
        Assign one or more categorical values to the array.

        This method mirrors NumPy / pandas-style semantics at the *value* level
        (category labels), not at the internal code level. Because Arkouda
        :class:`Categorical` objects are backed by immutable components, this
        implementation:

        1. Materializes the current values to a NumPy array of Python objects.
        2. Applies the requested mutation locally using NumPy's assignment
           semantics (including broadcasting and advanced indexing).
        3. Rebuilds a new Arkouda :class:`Categorical` from the mutated data.

        This is currently an ``O(n)`` operation in the array size.

        Parameters
        ----------
        key : int, slice, array-like, numpy.ndarray, or Arkouda indexer
            Location(s) to assign to. Supported forms include:

            * scalar integer index
            * slice objects
            * NumPy integer array (any integer dtype)
            * NumPy boolean mask with the same length as the array
            * Python list of integers or booleans
            * Arkouda ``pdarray`` of integers or booleans (via ``to_ndarray()``)

        value : str, array-like, ArkoudaCategoricalArray, or Arkouda array-like
            The value(s) to assign, interpreted as category *labels*.

            * A scalar Python ``str`` is broadcast to the selected positions.
            * An :class:`ArkoudaCategoricalArray` is converted to a NumPy array
              via :meth:`to_ndarray`.
            * Any Arkouda object with a :meth:`to_ndarray` method (e.g.
              :class:`arkouda.Categorical` or :class:`arkouda.Strings`) is
              converted via that method.
            * Other array-likes (lists of strings, NumPy arrays of dtype
              object/str, etc.) are used directly in NumPy assignment.

        Notes
        -----
        Categories are recomputed from the resulting values when the new
        Arkouda :class:`Categorical` is constructed. This means:

        * New categories introduced by assignment are automatically added.
        * Unused categories may disappear, mirroring a "re-categoricalize
          from values" behavior rather than in-place code mutation.

        Examples
        --------
        Basic scalar assignment by position:

        >>> import arkouda as ak
        >>> import numpy as np
        >>> from arkouda.pandas.extension import ArkoudaCategoricalArray
        >>> data = ak.Categorical(ak.array(["a", "b", "c"]))
        >>> arr = ArkoudaCategoricalArray(data)
        >>> arr[1] = "xx"
        >>> arr.to_ndarray()
        array(['a', 'xx', 'c'], dtype='<U3')

        Using a NumPy boolean mask:

        >>> data = ak.Categorical(ak.array(["a", "b", "c", "d", "e"]))
        >>> arr = ArkoudaCategoricalArray(data)
        >>> mask = np.array([True, False, True, False, True])
        >>> arr[mask] = "z"
        >>> arr.to_ndarray()
        array(['z', 'b', 'z', 'd', 'z'], dtype='<U3')

        Using a NumPy integer indexer with multiple values:

        >>> data = ak.Categorical(ak.array(["a", "b", "c", "d"]))
        >>> arr = ArkoudaCategoricalArray(data)
        >>> idx = np.array([0, 2], dtype=np.int64)
        >>> arr[idx] = ["x", "y"]
        >>> arr.to_ndarray()
        array(['x', 'b', 'y', 'd'], dtype='<U3')
        """
        import numpy as np

        from arkouda.numpy.pdarraycreation import array as ak_array

        # Work on a local NumPy *object-dtype* copy of the values. Using
        # object dtype avoids fixed-width Unicode truncation issues.
        base = self.to_ndarray()
        np_data = np.array(base.tolist(), dtype=object)

        # Normalize Arkouda indexers (e.g., pdarray) to NumPy, if present.
        if hasattr(key, "to_ndarray") and not isinstance(key, np.ndarray):
            key = key.to_ndarray()

        # Normalize the value into something NumPy can assign directly.
        if isinstance(value, ArkoudaCategoricalArray):
            value = value.to_ndarray()
        elif hasattr(value, "to_ndarray") and not isinstance(value, np.ndarray):
            # Covers arkouda.Categorical, arkouda.Strings, etc.
            value = value.to_ndarray()

        # Let NumPy handle broadcasting / advanced indexing semantics.
        np_data[key] = value

        # Rebuild an Arkouda Categorical from the mutated values.
        # 1. Convert the Python/NumPy object array to an Arkouda Strings.
        # 2. Build a new Categorical from those values.
        strings = ak_array(np_data)
        self._data = ak.Categorical(strings)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        from arkouda import Categorical, array

        # if 'scalars' are raw labels (strings), build ak.Categorical
        if not isinstance(scalars, Categorical):
            scalars = Categorical(array(scalars))
        return cls(scalars)

    def astype(self, x, dtype):
        raise NotImplementedError("array_api.astype is not implemented in Arkouda yet")

    def isna(self):
        return ak.zeros(self._data.size, dtype=ak.bool)

    @property
    def dtype(self):
        return ArkoudaCategoricalDtype()

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaCategoricalArray) else other)

    def __repr__(self):
        return f"ArkoudaCategoricalArray({self._data})"
