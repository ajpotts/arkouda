from typing import TYPE_CHECKING, TypeVar

from pandas.api.extensions import ExtensionArray

import arkouda as ak

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
        """
        Initialize an Arkouda-backed ExtensionArray for categorical data.

        This constructor ensures that the underlying data is an Arkouda
        `Categorical` object, representing a distributed categorical array
        stored on the Arkouda server.

        Parameters
        ----------
        data : arkouda.Categorical, ArkoudaCategoricalArray, list, tuple, or numpy.ndarray
            Input data to wrap or convert.

            - If `data` is an Arkouda `Categorical`, it is used directly.
            - If `data` is another `ArkoudaCategoricalArray`, its backing
              `Categorical` is reused.
            - If `data` is a list, tuple, or NumPy array, it is converted into
              an Arkouda `Categorical` via `ak.Categorical(ak.array(data))`.

        Raises
        ------
        TypeError
            If `data` cannot be converted to an Arkouda `Categorical`.

        Notes
        -----
        Arkouda categorical arrays store both the label and code arrays on
        the server and are optimized for efficient distributed groupby and
        factorization operations.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaCategoricalArray
        >>> c = ak.Categorical(ak.array(["red", "green", "blue"]))
        >>> ArkoudaCategoricalArray(c)
        ArkoudaCategoricalArray(['red', 'green', 'blue'])

        >>> ArkoudaCategoricalArray(["apple", "banana", "apple"])
        ArkoudaCategoricalArray(['apple', 'banana', 'apple'])
        """
        from arkouda import Categorical, array

        # Reuse backing categorical
        if isinstance(data, ArkoudaCategoricalArray):
            self._data = data._data
            return

        # Convert if given a Python/NumPy sequence
        if not isinstance(data, Categorical):
            try:
                data = Categorical(array(data))
            except Exception as e:
                raise TypeError(
                    f"Expected arkouda.Categorical or sequence convertible to one, got {type(data).__name__}"
                ) from e

        self._data = data

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        from arkouda import Categorical, array

        # if 'scalars' are raw labels (strings), build ak.Categorical
        if not isinstance(scalars, Categorical):
            scalars = Categorical(array(scalars))
        return cls(scalars)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._data[idx]
        return ArkoudaCategoricalArray(self._data[idx])

    def astype(self, x, dtype):
        raise NotImplementedError("array_api.astype is not implemented in Arkouda yet")

    def isna(self):
        return ak.zeros(self._data.size, dtype=ak.bool)

    def copy(self):
        return ArkoudaCategoricalArray(self._data[:])

    @property
    def dtype(self):
        return ArkoudaCategoricalDtype()

    def __eq__(self, other):
        return self._data == (other._data if isinstance(other, ArkoudaCategoricalArray) else other)

    def __repr__(self):
        return f"ArkoudaCategoricalArray({self._data})"
