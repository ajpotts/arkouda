"""
Wrapper class around the pdarray object for the array API standard.

The information below can be found online here:

https://pydoc.dev/numpy/latest/numpy.array_api._array_object.html

The array API standard defines some behaviors differently than ndarray, in
particular, type promotion rules are different (the standard has no
value-based casting). The standard also specifies a more limited subset of
array methods and functionalities than are implemented on ndarray. Since the
goal of the array_api namespace is to be a minimal implementation of the array
API standard, we need to define a separate wrapper class for the array_api
namespace.

The standard compliant class is only a wrapper class. It is *not* a subclass
of ndarray.
"""

from __future__ import annotations

from enum import IntEnum
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np

import arkouda as ak
from arkouda import array_api

from ._dtypes import (
    _complex_floating_dtypes,
    _dtype_categories,
    _floating_dtypes,
    _integer_dtypes,
    _result_type,
)
from ._dtypes import _boolean_dtypes  # _all_dtypes,; _integer_or_boolean_dtypes,; _numeric_dtypes,
from .creation_functions import asarray


if TYPE_CHECKING:
    from ._typing import Device, Dtype


HANDLED_FUNCTIONS: Dict[str, Callable] = {}


def _all_int_indices(key) -> bool:
    import numpy as np

    if not isinstance(key, tuple):
        key = (key,)
    for k in key:
        if k is Ellipsis:
            continue
        if isinstance(k, slice):
            return False
        if isinstance(k, (list, tuple, np.ndarray)):
            return False
        if hasattr(k, "shape"):  # Array or np scalar array
            # only OK if it's a scalar
            try:
                size = k.size
                shape = k.shape
            except Exception:
                return False
            if not (size == 1 or shape == ()):
                return False
        elif not isinstance(k, (int, np.integer)):
            return False
    return True


def _wrap_scalar(value, dt) -> "Array":
    import arkouda as ak

    ndt = _normalize_dt(dt)  # np.dtype
    pyv = _to_builtin_for_dtype(value, ndt)
    return Array._new(ak.full(1, pyv, dtype=ndt))


def _promote2(lhs_dt, rhs_dt):
    # lhs_dt/rhs_dt may be str/np.dtype/np type — normalize both.
    return np.result_type(_normalize_dt(lhs_dt), _normalize_dt(rhs_dt))


def _wrap_scalar_with_dtype(val, res_dt) -> "Array":
    """
    Robust scalar wrapper: allocate on the server with ak.full to avoid host-side
    NumPy packing issues that can corrupt float payloads.
    """
    v = _to_python_builtin(val)
    dt = _normalize_dtype_for_ak_array(res_dt)
    # IMPORTANT: use ak.full, not ak.array([...])
    return Array._new(ak.full(1, v, dtype=dt))


def _normalize_dt(dt):
    import numpy as np

    # Always return a *np.dtype* object
    try:
        return np.dtype(dt)
    except TypeError:
        # last-ditch: objects that carry a dtype-like
        inner = getattr(dt, "dtype", None)
        if inner is not None:
            return np.dtype(inner)
        raise


def _normalize_dtype_for_ak_array(dt):
    import numpy as np

    if isinstance(dt, np.dtype):
        return dt
    if isinstance(dt, type) and getattr(dt, "__module__", "") == np.__name__:
        return dt
    if isinstance(dt, str):
        try:
            return np.dtype(dt)
        except Exception:
            pass
    return np.dtype(dt)


def _to_builtin_for_dtype(val, dt):
    import numpy as np

    if isinstance(val, np.generic):
        val = val.item()
    k = dt.kind  # now safe: dt is a np.dtype
    if k == "f":
        return float(val)
    if k in ("i", "u"):
        return int(val)
    if k == "b":
        return bool(val)
    return float(val)  # fallback


def _to_python_builtin(val):
    import numpy as np

    return val.item() if isinstance(val, np.generic) else val


def _is_scalar_array(x: "Array") -> bool:
    return isinstance(x, Array) and (x.shape == () or x.size == 1)


def _to_python_scalar(x: "Array"):
    a = x._array
    return a[0] if getattr(a, "size", 1) == 1 else a  # returns NumPy/Python scalar


# array_object.py
def scalar_array(x, dtype: str | None = None) -> ak.pdarray:
    """
    Wrap a Python/NumPy scalar as a 0-D Array (backed by a 1-element pdarray).
    Correctly infers dtype so floats don't get packed as ints, etc.
    """
    import numpy as np

    import arkouda as ak

    # If user explicitly provided dtype, trust it
    if dtype is None:
        # NumPy scalars carry a dtype already
        if isinstance(x, np.generic):
            kind = x.dtype.kind
            if kind == "b":
                dtype = "bool"
            elif kind == "f":
                dtype = "float64"
            elif kind == "i":
                dtype = "int64"
            elif kind == "u":
                # Array API doesn’t define unsigned scalars; pick int64 unless you need uint64
                # If your semantics prefer uint64 here, set dtype = "uint64"
                dtype = "int64"
            else:
                # Fallback: float64 is safest
                dtype = "float64"
        else:
            # Plain Python types
            if isinstance(x, bool):
                dtype = "bool"
            elif isinstance(x, float):
                dtype = "float64"
            elif isinstance(x, int):
                # If you want magnitude-aware routing, you could call ak.numpy.dtypes.dtype(x)
                # For Array API consistency, int64 is fine:
                dtype = "int64"
            else:
                # Last resort: try float
                dtype = "float64"

    # Build a 1-element array then treat it as 0-D at the wrapper layer
    return ak.array([x], dtype=dtype)


class Array:
    """
    n-dimensional array object for the array API namespace.

    See the docstring of :py:obj:`np.ndarray <numpy.ndarray>` for more
    information.

    This is a wrapper around ak.pdarray that restricts the usage to only
    those things that are required by the array API namespace. Note,
    attributes on this object that start with a single underscore are not part
    of the API specification and should only be used internally. This object
    should not be constructed directly. Rather, use one of the creation
    functions, such as asarray().

    """

    _array: ak.pdarray
    _empty: bool

    # Use a custom constructor instead of __init__, as manually initializing
    # this class is not supported by the API.
    @classmethod
    def _new(cls, x, /, empty: bool = False):
        """
        Initialize the array API Array object.

        Functions outside of the array_api submodule should not use this
        method. Use one of the creation functions instead, such as
        ``asarray``.

        """
        obj = super().__new__(cls)
        obj._array = x
        obj._empty = empty
        return obj

    # Prevent Array() from working
    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "The array_api Array object should not be instantiated directly. \
            Use an array creation function, such as asarray(), instead."
        )

    def tolist(self):
        """
        Convert the array to a Python list or nested lists, using the pdarray
        method tolist.

        This involves copying the data from the server to the client, and thus
        will fail if the array is too large (see:
        :func:`~arkouda.client.maxTransferBytes`)

        See Also
        --------
        pdarray.tolist()

        """
        x = self._array.tolist()
        if self.shape == ():
            # to match numpy, return a scalar for a 0-dimensional array
            return x[0]
        else:
            return x

    def to_ndarray(self):
        """
        Convert the array to a numpy ndarray, using the pdarray method to_ndarray.

        This involves copying the data from the server to the client, and thus
        will fail if the array is too large (see:
        :func:`~arkouda.client.maxTransferBytes`)

        See Also
        --------
        pdarray.to_ndarray()

        """
        return self._array.to_ndarray()

    def item(self):
        """
        Get the scalar value from a 0-dimensional array.

        Raises a ValueError if the array has more than one element.
        """
        if self._has_single_elem():
            return self._array[0]
        else:
            raise ValueError("Can only convert an array with one element to a Python scalar")

    def transpose(self, axes: Optional[Tuple[int, ...]] = None):
        """
        Return a view of the array with the specified axes transposed.

        For axes=None, reverse all the dimensions of the array.

        See Also
        --------
        ak.transpose()

        """
        return asarray(ak.transpose(self._array, axes))

    def __str__(self: Array, /) -> str:
        """Perform the operation __str__."""
        return self._array.__str__()

    def __repr__(self: Array, /) -> str:
        """Perform the operation __repr__."""
        return f"Arkouda Array ({self.shape}, {self.dtype})" + self._array.__str__()

    def _repr_inline_(self: Array, width: int) -> str:
        """
        Get a single line representation of the array for display in a space
        constrained context like a Jupyter notebook cell.
        """
        return f"Arkouda Array ({self.shape}, {self.dtype})"

    def chunk_info(self: Array, /) -> List[List[int]]:
        """
        Get a list of indices indicating how the array is chunked across
        Locales (compute nodes). Although Arkouda arrays don't have a notion
        of chunking, like Dask arrays for example, it can be useful to know
        how the array is distributed across locales in order to write/read
        data to/from a chunked format like Zarr.

        Returns a nested list of integers, where the outer list corresponds to
        dimensions, and the inner lists correspond to locales. The value at [d][l]
        is the global array index where locale l's local subdomain along the
        d-th dimension begins.

        For example, calling this function on a 100x40 2D array stored across 4
        locales could return: [[0, 50], [0, 20]], indicating that the 4 "chunks"
        start at indices 0 and 50 in the first dimension, and 0 and 20 in the
        second dimension.
        """
        import json

        def extract_chunk(msg_str: str):
            return "".join(msg_str.split()[1:])

        return json.loads(
            extract_chunk(
                cast(
                    str,
                    ak.client.generic_msg(
                        cmd=f"chunkInfoAsString<{self.dtype},{self.ndim}>",
                        args={"array": self._array},
                    ),
                )  # string returned has format "str {list of lists}"
            )
        )

    def __array__(self, dtype: None | np.dtype[Any] = None):
        """Get a numpy ndarray."""
        return np.asarray(self.to_ndarray(), dtype=dtype)

    __array_ufunc__ = None

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def _check_allowed_dtypes(
        self, other: bool | int | float | Array, dtype_category: str, op: str
    ) -> Array:
        """
        Allow only specific input dtypes.

        Helper function for operators.

        Use like

            other = self._check_allowed_dtypes(other, 'numeric', '__add__')
            if other is NotImplemented:
                return other
        """
        if self.dtype not in _dtype_categories[dtype_category]:
            raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        if isinstance(other, (int, complex, float, bool)):
            other = self._promote_scalar(other)
        elif isinstance(other, Array):
            if other.dtype not in _dtype_categories[dtype_category]:
                raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        else:
            return NotImplemented

        assert isinstance(other, Array)

        # This will raise TypeError for type combinations that are not allowed
        # to promote in the spec (even if the NumPy array operator would
        # promote them).
        res_dtype = _result_type(self.dtype, other.dtype)
        if op.startswith("__i"):
            # Note: NumPy will allow in-place operators in some cases where
            # the type promoted operator does not match the left-hand side
            # operand. For example,

            # >>> a = np.array(1, dtype=np.int8)
            # >>> a += np.array(1, dtype=np.int16)

            # The spec explicitly disallows this.
            if res_dtype != self.dtype:
                raise TypeError(f"Cannot perform {op} with dtypes {self.dtype} and {other.dtype}")

        return other

    # Helper function to match the type promotion rules in the spec
    def _promote_scalar(self, scalar) -> Array:
        """
        Return a promoted version of a Python scalar appropriate for use with
        operations on self.

        This may raise an OverflowError in cases where the scalar is an
        integer that is too large to fit in a NumPy integer dtype, or
        TypeError when the scalar type is incompatible with the dtype of self.
        """
        # Note: Only Python scalar types that match the array dtype are
        # allowed.
        if isinstance(scalar, bool):
            if self.dtype not in _boolean_dtypes:
                raise TypeError("Python bool scalars can only be promoted with bool arrays")
        elif isinstance(scalar, int):
            if self.dtype in _boolean_dtypes:
                raise TypeError("Python int scalars cannot be promoted with bool arrays")
            if self.dtype in _integer_dtypes:
                info = np.iinfo(int)
                if not (info.min <= scalar <= info.max):
                    raise OverflowError(
                        "Python int scalars must be within the bounds of the dtype for integer arrays"
                    )
            # int + array(floating) is allowed
        elif isinstance(scalar, float):
            if self.dtype not in _floating_dtypes:
                raise TypeError("Python float scalars can only be promoted with floating-point arrays.")
        elif isinstance(scalar, complex):
            if self.dtype not in _complex_floating_dtypes:
                raise TypeError(
                    "Python complex scalars can only be promoted with complex floating-point arrays."
                )
        else:
            raise TypeError("'scalar' must be a Python scalar")

        # Note: scalars are unconditionally cast to the same dtype as the
        # array.

        # Note: the spec only specifies integer-dtype/int promotion
        # behavior for integers within the bounds of the integer dtype.
        # Outside of those bounds we use the default NumPy behavior (either
        # cast or raise OverflowError).
        return Array._new(np.array(scalar, self.dtype))

    @staticmethod
    def _normalize_two_args(x1, x2) -> Tuple[Array, Array]:
        """
        Normalize inputs to two arg functions to fix type promotion rules.

        NumPy deviates from the spec type promotion rules in cases where one
        argument is 0-dimensional and the other is not. For example:

        >>> import numpy as np
        >>> a = np.array([1.0], dtype=np.float32)
        >>> b = np.array(1.0, dtype=np.float64)
        >>> np.add(a, b).dtype # The spec says this should be float64
        dtype('float64')

        To fix this, we add a dimension to the 0-dimension array before passing it
        through. This works because a dimension would be added anyway from
        broadcasting, so the resulting shape is the same, but this prevents NumPy
        from not promoting the dtype.
        """
        # Another option would be to use signature=(x1.dtype, x2.dtype, None),
        # but that only works for ufuncs, so we would have to call the ufuncs
        # directly in the operator methods. One should also note that this
        # sort of trick wouldn't work for functions like searchsorted, which
        # don't do normal broadcasting, but there aren't any functions like
        # that in the array API namespace.
        if x1.ndim == 0 and x2.ndim != 0:
            # The _array[None] workaround was chosen because it is relatively
            # performant. broadcast_to(x1._array, x2.shape) is much slower. We
            # could also manually type promote x2, but that is more complicated
            # and about the same performance as this.
            x1 = Array._new(x1._array[None])
        elif x2.ndim == 0 and x1.ndim != 0:
            x2 = Array._new(x2._array[None])
        return (x1, x2)

    # Note: A large fraction of allowed indices are disallowed here (see the
    # docstring below)
    def _validate_index(self, key):
        raise IndexError("not implemented")

    def __abs__(self: Array, /) -> Array:
        """
        Take the element-wise absolute value of the array.

        See Also
        --------
        ak.abs()

        """
        return Array._new(ak.abs(self._array))

    def __add__(self, other):
        if _is_scalar_array(self) and _is_scalar_array(other):
            lhs = _to_python_scalar(self)
            rhs = _to_python_scalar(other)
            res_dt = _promote2(self.dtype, other.dtype)
            return _wrap_scalar(lhs + rhs, res_dt)
        return Array._new(self._array + (other._array if isinstance(other, Array) else other))

    def __and__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical AND operation of this array and another array or scalar."""
        if isinstance(other, (int, float)):
            return Array._new(self._array and other)
        else:
            return Array._new(self._array and other._array)

    def __array_namespace__(self: Array, /, *, api_version: Optional[str] = None) -> types.ModuleType:
        """Get the array API namespace from an `Array` instance."""
        if api_version is not None:
            raise ValueError(f"Unrecognized array API version: {api_version!r}")
        return array_api

    def __bool__(self: Array, /) -> bool:
        """Get the truth value of a single element array."""
        s = self._single_elem()
        if s is not None:
            return bool(s)
        else:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. Use 'any' or 'all'"
            )

    def __complex__(self: Array, /) -> complex:
        """Get a complex value from a single element array."""
        if s := self._single_elem():
            return complex(s)
        else:
            raise ValueError("cannot convert non-scalar array to complex")

    def __dlpack_device__(self: Array, /) -> Tuple[IntEnum, int]:
        """
        Returns device type and device ID in DLPack format.

        Warning: Not implemented.
        """
        raise ValueError("Not implemented")

    def __eq__(self: Array, other: object, /) -> bool:
        """Check if this array is equal to another array or scalar."""
        if isinstance(other, (int, bool, float)):
            return self._array == scalar_array(other)
        elif isinstance(other, Array):
            return self._array == other._array
        else:
            return False

    def __float__(self: Array, /) -> float:
        """Get a float value from a single element array."""
        if s := self._single_elem():
            if isinstance(s, complex):
                raise TypeError("can't convert complex to float")
            else:
                return float(s)
        else:
            raise ValueError("cannot convert non-scalar array to float")

    def __floordiv__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the floor division of this array by another array or scalar."""
        if isinstance(other, (int, float)):
            return Array._new(self._array // other)
        else:
            return Array._new(self._array // other._array)

    def __ge__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Check if this array is greater than or equal to another array or scalar."""
        if isinstance(other, (int, float)):
            return Array._new(self._array >= other)
        else:
            return Array._new(self._array >= other._array)

    def __getitem__(self, key) -> "Array":
        import numpy as np

        def _unwrap(kt):
            from .array_object import asarray

            if isinstance(kt, Array):
                return kt._array[0] if (kt.size == 1 or kt.shape == ()) else kt._array
            if isinstance(kt, np.ndarray):
                a = asarray(kt)
                return a._array[0] if (a.size == 1 or a.shape == ()) else a._array
            return kt

        orig = key
        if isinstance(key, tuple):
            k = tuple(_unwrap(kt) for kt in key)
        else:
            k = _unwrap(key)

        a = self._array[k]

        # a is result of self._array[k]
        if isinstance(a, ak.pdarray):
            if getattr(a, "size", 1) == 1:
                return _wrap_scalar(a[0], a.dtype)  # use the pdarray’s dtype
            return Array._new(a)
        # Python scalar (0-d selection on arkouda server gave us a Python value)
        return _wrap_scalar(a, self.dtype)  # indexing does not change dtype

    def __gt__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Check if this array is greater than another array or scalar."""
        if isinstance(other, (int, float)):
            return Array._new(self._array > other)
        else:
            return Array._new(self._array > other._array)

    def __int__(self: Array, /) -> int:
        """Get an integer value from a single element array."""
        if s := self._single_elem():
            if isinstance(s, complex):
                raise TypeError("can't convert complex to int")
            else:
                return int(s)
        else:
            raise ValueError("cannot convert non-scalar array to int")

    def __index__(self: Array, /) -> int:
        """Get an integer value from a single element array."""
        if s := self._single_elem():
            if isinstance(s, int):
                return s
            else:
                raise TypeError("Only integer arrays can be converted to a Python integer")
        else:
            raise ValueError("cannot convert non-scalar array to int")

    def __invert__(self: Array, /) -> Array:
        """Compute the logical NOT operation on this array."""
        if self.dtype in _integer_dtypes or self.dtype in _boolean_dtypes:
            return Array._new(~self._array)
        else:
            raise TypeError("Only integer and boolean arrays can be inverted")

    def __le__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Check if this array is less than or equal to another array or scalar."""
        if isinstance(other, (int, float)):
            return Array._new(self._array <= other)
        else:
            return Array._new(self._array <= other._array)

    def __lshift__(self: Array, other: Union[int, Array], /) -> Array:
        """Compute the left shift of this array by another array or scalar."""
        if isinstance(other, int):
            return Array._new(self._array << other)
        else:
            return Array._new(self._array << other._array)

    def __lt__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Check if this array is less than another array or scalar."""
        if isinstance(other, (int, float)):
            return Array._new(self._array < other)
        else:
            return Array._new(self._array < other._array)

    def __matmul__(self: Array, other: Array, /) -> Array:
        """
        Compute the matrix multiplication of this array with another array.

        See Also
        --------
        ak.matmul()
        """
        #       raise ValueError("Not implemented")
        return asarray(ak.matmul(self._array, other._array))

    def __mod__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the modulo of this array by another array or scalar."""
        if isinstance(other, (int, float)):
            return Array._new(self._array % other)
        else:
            return Array._new(self._array % other._array)

    def __mul__(self, other):
        if _is_scalar_array(self) and _is_scalar_array(other):
            lhs = _to_python_scalar(self)
            rhs = _to_python_scalar(other)
            res_dt = _promote2(self.dtype, other.dtype)
            return _wrap_scalar(lhs * rhs, res_dt)
        return Array._new(self._array * (other._array if isinstance(other, Array) else other))

    def __ne__(self: Array, other: object, /) -> bool:
        """Check if this array is not equal to another array or scalar."""
        if isinstance(other, (int, bool, float)):
            return self._array != scalar_array(other)
        elif isinstance(other, Array):
            return self._array != other._array
        else:
            return False

    def __neg__(self: Array, /) -> Array:
        """Compute the element-wise negation of this array."""
        return Array._new(-self._array)

    def __or__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical OR operation of this array and another array or scalar."""
        if isinstance(other, (int, bool)):
            return Array._new(self._array or other)
        else:
            return Array._new(self._array or other._array)

    def __pos__(self: Array, /) -> Array:
        """Compute the element-wise positive of this array."""
        return self

    def __pow__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the power of this array by another array or scalar."""
        if isinstance(other, (int, float)):
            return Array._new(self._array**other)
        else:
            return Array._new(self._array**other._array)

    def __rshift__(self: Array, other: Union[int, Array], /) -> Array:
        """Compute the right shift of this array by another array or scalar."""
        if isinstance(other, int):
            return Array._new(self._array >> other)
        else:
            return Array._new(self._array >> other._array)

    def __setitem__(
        self,
        key: Union[int, slice, Tuple[Union[int, slice], ...], Array],
        value: Union[int, float, bool, Array],
        /,
    ) -> None:
        if isinstance(key, Array):
            if isinstance(value, Array):
                if value.size == 1 or value.shape == ():
                    self._array[key._array] = value._array[0]
                else:
                    self._array[key._array] = value._array
            else:
                self._array[key._array] = value
        else:
            if isinstance(value, Array):
                if value.size == 1 or value.shape == ():
                    self._array[key] = value._array[0]
                else:
                    self._array[key] = value._array
            else:
                self._array[key] = value

    def __sub__(self, other):
        if _is_scalar_array(self) and _is_scalar_array(other):
            lhs = _to_python_scalar(self)
            rhs = _to_python_scalar(other)
            res_dt = _promote2(self.dtype, other.dtype)
            return _wrap_scalar(lhs - rhs, res_dt)
        return Array._new(self._array - (other._array if isinstance(other, Array) else other))

    def __truediv__(self, other):
        if _is_scalar_array(self) and _is_scalar_array(other):
            lhs = _to_python_scalar(self)
            rhs = _to_python_scalar(other)
            return _wrap_scalar(lhs / rhs, np.float64)  # Array API: division -> float
        return Array._new(self._array / (other._array if isinstance(other, Array) else other))

    def __xor__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical XOR operation of this array and another array or scalar."""
        if isinstance(other, (int, bool)):
            return Array._new(self._array ^ other)
        else:
            return Array._new(self._array ^ other._array)

    def __iadd__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the sum of this array and another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array += other
            return self
        else:
            self._array += other._array
            return self

    def __radd__(self, other):
        if _is_scalar_array(self) and _is_scalar_array(other):
            rhs = _to_python_scalar(self)
            lhs = _to_python_scalar(other)
            res_dt = _promote2(other.dtype, self.dtype)
            return _wrap_scalar(lhs + rhs, res_dt)
        return Array._new((other._array if isinstance(other, Array) else other) + self._array)

    def __iand__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical AND operation of this array and another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array &= other
            return self
        else:
            self._array &= other._array
            return self

    def __rand__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical AND operation of another array or scalar and this array."""
        if isinstance(other, (int, float)):
            return Array._new(other and self._array)
        else:
            return Array._new(other._array and self._array)

    def __ifloordiv__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the floor division of this array by another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array //= other
            return self
        else:
            self._array //= other._array
            return self

    def __rfloordiv__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the floor division of another array or scalar by this array."""
        if isinstance(other, (int, float)):
            return Array._new(other // self._array)
        else:
            return Array._new(other._array // self._array)

    def __ilshift__(self: Array, other: Union[int, Array], /) -> Array:
        """Compute the left shift of this array by another array or scalar in place."""
        if isinstance(other, int):
            self._array <<= other
            return self
        else:
            self._array <<= other._array
            return self

    def __rlshift__(self: Array, other: Union[int, Array], /) -> Array:
        """Compute the left shift of another array or scalar by this array."""
        if isinstance(other, int):
            return Array._new(other << self._array)
        else:
            return Array._new(other._array << self._array)

    def __imatmul__(self: Array, other: Array, /) -> Array:
        """
        Compute the matrix multiplication of this array with another array in place.

        Warning: Not implemented.
        """
        raise ValueError("Not implemented")

    def __rmatmul__(self: Array, other: Array, /) -> Array:
        """
        Compute the matrix multiplication of another array with this array.

        Warning: Not implemented.
        """
        raise ValueError("Not implemented")

    def __imod__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the modulo of this array by another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array %= other
            return self
        else:
            self._array %= other._array
            return self

    def __rmod__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the modulo of another array or scalar by this array."""
        if isinstance(other, (int, float)):
            return Array._new(other % self._array)
        else:
            return Array._new(other._array % self._array)

    def __imul__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the product of this array and another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array *= other
            return self
        else:
            self._array *= other._array
            return self

    def __rmul__(self, other):
        if _is_scalar_array(self) and _is_scalar_array(other):
            rhs = _to_python_scalar(self)
            lhs = _to_python_scalar(other)
            res_dt = _promote2(other.dtype, self.dtype)
            return _wrap_scalar(lhs * rhs, res_dt)
        return Array._new((other._array if isinstance(other, Array) else other) * self._array)

    def __ior__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical OR operation of this array and another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array |= other
            return self
        else:
            self._array |= other._array
            return self

    def __ror__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical OR operation of another array or scalar and this array."""
        if isinstance(other, (int, float)):
            return Array._new(other or self._array)
        else:
            return Array._new(other._array or self._array)

    def __ipow__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the power of this array by another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array **= other
            return self
        else:
            self._array **= other._array
            return self

    def __rpow__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the power of another array or scalar by this array."""
        if isinstance(other, (int, float)):
            return Array._new(other**self._array)
        else:
            return Array._new(other._array**self._array)

    def __irshift__(self: Array, other: Union[int, Array], /) -> Array:
        """Compute the right shift of this array by another array or scalar in place."""
        if isinstance(other, int):
            self._array >>= other
            return self
        else:
            self._array >>= other._array
            return self

    def __rrshift__(self: Array, other: Union[int, Array], /) -> Array:
        """Compute the right shift of another array or scalar by this array."""
        if isinstance(other, int):
            return Array._new(other >> self._array)
        else:
            return Array._new(other._array >> self._array)

    def __isub__(self: Array, other: Union[int, float, Array], /) -> Array:
        """Compute the difference of this array and another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array -= other
            return self
        else:
            self._array -= other._array
            return self

    def __rsub__(self, other):
        if _is_scalar_array(self) and _is_scalar_array(other):
            rhs = _to_python_scalar(self)
            lhs = _to_python_scalar(other)
            res_dt = _promote2(other.dtype, self.dtype)
            return _wrap_scalar(lhs - rhs, res_dt)
        return Array._new((other._array if isinstance(other, Array) else other) - self._array)

    def __itruediv__(self: Array, other: Union[float, Array], /) -> Array:
        """Compute the true division of this array by another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array /= other
            return self
        else:
            self._array /= other._array
            return self

    def __rtruediv__(self, other):
        if _is_scalar_array(self) and _is_scalar_array(other):
            rhs = _to_python_scalar(self)
            lhs = _to_python_scalar(other)
            return _wrap_scalar(lhs / rhs, np.float64)
        return Array._new((other._array if isinstance(other, Array) else other) / self._array)

    def __ixor__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical XOR operation of this array and another array or scalar in place."""
        if isinstance(other, (int, float)):
            self._array ^= other
            return self
        else:
            self._array ^= other._array
            return self

    def __rxor__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """Compute the logical XOR operation of another array or scalar by this array."""
        if isinstance(other, (int, float)):
            return Array._new(other ^ self._array)
        else:
            return Array._new(other._array ^ self._array)

    def to_device(self: Array, device: Device, /, stream: None = None) -> Array:
        raise ValueError("Not implemented")

    def _has_single_elem(self: Array, /) -> bool:
        return self._array.shape == [] or self._array.size == 1

    def _single_elem(self: Array) -> Optional[Union[int, float, complex, bool]]:
        if self._has_single_elem():
            if self.ndim > 0:
                return self._array[(0,) * self.ndim]
            else:
                return self._array[0]
        else:
            return None

    @property
    def dtype(self) -> Dtype:
        return self._array.dtype

    @property
    def device(self) -> Device:
        return "cpu"

    @property
    def mT(self) -> Array:
        raise ValueError("Not implemented")

    @property
    def ndim(self) -> int:
        # note: this is not the same as 'self._array.ndim'
        # because 0D/scalar pdarrays will have ndim=1
        # but have a shape of '()'
        return len(self._array.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def size(self) -> int:
        return int(self._array.size)

    @property
    def T(self) -> Array:
        raise ValueError("Not implemented")


def implements_numpy(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator
