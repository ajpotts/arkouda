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
        if isinstance(data, np.ndarray):
            from arkouda.numpy.pdarraycreation import array as ak_array

            data = ak_array(data)

        if isinstance(data, ArkoudaStringArray):
            self._data = data._data
        elif isinstance(data, Strings):
            self._data = data
        else:
            raise TypeError(f"Expected arkouda Strings. Instead received {type(data)}.")

    @property
    def dtype(self):
        return ArkoudaStringDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(ak_array(scalars))

    def __getitem__(self, key):
        """
        Retrieve one or more string values.

        Parameters
        ----------
        key : int, slice, list, numpy.ndarray, or Arkouda indexer
            Positional indexer. Supports:
            * scalar integer positions
            * slice objects
            * NumPy integer arrays (signed/unsigned)
            * NumPy boolean masks
            * Python lists of integers / booleans
            * Arkouda pdarray indexers (int / uint / bool)

        Returns
        -------
        str or ArkoudaStringArray
            A Python string for scalar access, or a new ArkoudaStringArray
            for non-scalar indexers.
        """
        # Normalize NumPy indexers to Arkouda pdarrays, mirroring ArkoudaArray.__getitem__
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = ak_array(key)
            elif key.dtype.kind in {"i"}:
                # signed integer
                key = ak_array(key, dtype="int64")
            elif key.dtype.kind in {"u"}:
                # unsigned integer
                key = ak_array(key, dtype="uint64")
            else:
                raise TypeError(f"Unsupported numpy index type {key.dtype}")

        result = self._data[key]

        # Scalar access: return a plain Python str (or scalar) instead of a Strings object
        if np.isscalar(key):
            # Arkouda may hand back a tiny array-like; normalize via to_ndarray if available
            if hasattr(result, "to_ndarray"):
                arr = result.to_ndarray()
                # Support both 0-d and length-1 arrays
                try:
                    return arr[()]
                except Exception:
                    return arr[0]
            return result

        # Non-scalar: expect an Arkouda Strings, wrap it
        if isinstance(result, Strings):
            return ArkoudaStringArray(result)

        # Fallback: if Arkouda returned something array-like but not Strings,
        # materialize via ak.array and wrap again as Strings.
        return ArkoudaStringArray(ak_array(result))

    def __setitem__(self, key, value) -> None:
        """
        Assign one or more string values to the array.

        This method mirrors NumPy / pandas-style semantics by materializing the
        underlying Arkouda :class:`Strings` to a local NumPy array, performing
        the mutation locally, and then recreating a new :class:`Strings`
        instance. This is currently an O(n) operation in the array size.

        Parameters
        ----------
        key : int, slice, array-like, or numpy.ndarray
            Location(s) to assign to. Supported forms include:

            * scalar integer index
            * slice objects
            * NumPy integer array (any integer dtype)
            * NumPy boolean mask with the same length as the array
            * Python list of integers or booleans
            * Arkouda ``pdarray`` (integer or boolean)

            Any Arkouda indexers are converted to NumPy via ``to_ndarray()``
            before applying the mutation.

        value : str, array-like, ArkoudaStringArray, or arkouda.numpy.strings.Strings
            The value(s) to assign.

            * A scalar Python ``str`` is broadcast to the selected locations.
            * An :class:`ArkoudaStringArray` is converted to a NumPy array via
              :meth:`to_ndarray`.
            * A raw Arkouda :class:`Strings` object is converted via its
              :meth:`to_ndarray`.
            * Any other array-like of strings is used directly in NumPy
              assignment.

        Notes
        -----
        Unlike the numeric :class:`ArkoudaArray` implementation, this method
        does **not** perform in-place mutation on the server, because
        :class:`Strings` is immutable. Instead it performs a local copy–modify–
        rebuild cycle::

            np_data = self.to_ndarray()
            np_data[key] = value
            self._data = ak_array(np_data)

        To avoid NumPy's fixed-width Unicode truncation (e.g. silently
        truncating ``"xx"`` to ``"x"`` when the original dtype is ``'<U1'``),
        the data is first upcast to an object-dtype array.

        Examples
        --------
        Basic scalar assignment by position:

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaStringArray
        >>> data = ak.array(["a", "b", "c"])
        >>> arr = ArkoudaStringArray(data)
        >>> arr[1] = "xx"
        >>> arr.to_ndarray()
        array(['a', 'xx', 'c'], dtype='<U2')

        Using a NumPy boolean mask:

        >>> data = ak.array(["a", "b", "c", "d", "e"])
        >>> arr = ArkoudaStringArray(data)
        >>> mask = np.array([True, False, True, False, True])
        >>> arr[mask] = "z"
        >>> arr.to_ndarray()
        array(['z', 'b', 'z', 'd', 'z'], dtype='<U1')
        """
        import numpy as np

        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.numpy.strings import Strings

        # Work on a local NumPy *object-dtype* copy of the data to avoid
        # fixed-width Unicode truncation (e.g. '<U1').
        base = self.to_ndarray()
        np_data = np.array(base.tolist(), dtype=object)

        # Normalize Arkouda indexers to NumPy, if present
        if hasattr(key, "to_ndarray") and not isinstance(key, np.ndarray):
            key = key.to_ndarray()

        # Normalize value into something NumPy can assign directly
        if isinstance(value, ArkoudaStringArray):
            value = value.to_ndarray()
        elif isinstance(value, Strings):
            value = value.to_ndarray()

        # Let NumPy handle broadcasting / advanced indexing semantics
        np_data[key] = value

        # Rebuild an Arkouda Strings from the mutated NumPy array
        self._data = ak_array(np_data)

    def astype(self, dtype, copy: bool = False):
        if dtype in (object, np.object_, "object", np.dtype("O")):
            return self.to_ndarray().astype(object, copy=copy)
        # Let pandas do the rest locally
        return self.to_ndarray().astype(dtype, copy=copy)

    def isna(self):
        from arkouda.numpy.pdarraycreation import zeros

        return zeros(self._data.size, dtype="bool")

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaStringArray) else other)

    def __repr__(self):
        return f"ArkoudaStringArray({self._data})"
