from __future__ import annotations

import builtins
from enum import Enum
import sys
from typing import Union, cast

import numpy as np
from numpy import (
    bool,
    bool_,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    str_,
    uint8,
    uint16,
    uint32,
    uint64,
)
from numpy.dtypes import (
    BoolDType,
    ByteDType,
    BytesDType,
    CLongDoubleDType,
    Complex64DType,
    Complex128DType,
    DateTime64DType,
    Float16DType,
    Float32DType,
    Float64DType,
    Int8DType,
    Int16DType,
    Int32DType,
    Int64DType,
    IntDType,
    LongDoubleDType,
    LongDType,
    LongLongDType,
    ObjectDType,
    ShortDType,
    StrDType,
    TimeDelta64DType,
    UByteDType,
    UInt8DType,
    UInt16DType,
    UInt32DType,
    UInt64DType,
    UIntDType,
    ULongDType,
    ULongLongDType,
    UShortDType,
    VoidDType,
)


__all__ = [
    "_datatype_check",
    "ARKOUDA_SUPPORTED_DTYPES",
    "ARKOUDA_SUPPORTED_INTS",
    "DType",
    "DTypeObjects",
    "DTypes",
    "NUMBER_FORMAT_STRINGS",
    "NumericDTypes",
    "ScalarDTypes",
    "SeriesDTypes",
    "_is_dtype_in_union",
    "_val_isinstance_of_union",
    "all_scalars",
    "bigint",
    "bigint_",
    "bitType",
    "bool",
    "bool_scalars",
    "can_cast",
    "complex128",
    "complex64",
    "dtype",
    "dtype_for_chapel",
    "float16",
    "float32",
    "float64",
    "float_scalars",
    "get_byteorder",
    "get_server_byteorder",
    "int16",
    "int32",
    "int64",
    "int8",
    "intTypes",
    "int_scalars",
    "isSupportedBool",
    "isSupportedDType",
    "isSupportedFloat",
    "isSupportedInt",
    "isSupportedNumber",
    "numeric_and_bool_scalars",
    "numeric_scalars",
    "numpy_scalars",
    "resolve_scalar_dtype",
    "result_type",
    "str_",
    "str_scalars",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "BoolDType",
    "ByteDType",
    "BytesDType",
    "CLongDoubleDType",
    "Complex64DType",
    "Complex128DType",
    "DateTime64DType",
    "Float16DType",
    "Float32DType",
    "Float64DType",
    "Int8DType",
    "Int16DType",
    "Int32DType",
    "Int64DType",
    "IntDType",
    "LongDoubleDType",
    "LongDType",
    "LongLongDType",
    "ObjectDType",
    "ShortDType",
    "StrDType",
    "TimeDelta64DType",
    "UByteDType",
    "UInt8DType",
    "UInt16DType",
    "UInt32DType",
    "UInt64DType",
    "UIntDType",
    "ULongDType",
    "ULongLongDType",
    "UShortDType",
    "VoidDType",
]


NUMBER_FORMAT_STRINGS = {
    "bool": "{}",
    "int64": "{:d}",
    "float64": "{:.17f}",
    "uint8": "{:d}",
    "np.float64": "{f}",
    "uint64": "{:d}",
    "bigint": "{:d}",
}


def _datatype_check(the_dtype, allowed_list, name):
    if the_dtype not in allowed_list:
        raise TypeError(f"{name} only implements types {allowed_list}")


def dtype(x):
    """
    Normalize a dtype-like object or scalar/class into an Arkouda dtype sentinel
    or a NumPy dtype for non-Arkouda types.

    Rules:
      - "bigint" (string), bigint sentinel instance/class, or bigint_ scalar/class → ak.bigint()
      - Python ints routed by magnitude:
          [-2^63, 2^63-1] → int64
          [2^63, 2^64-1]  → uint64
          outside that     → ak.bigint()
      - Python float → float64
      - Python bool  → bool_
      - "str"/"str_" or str/np.str_ → np.str_
      - Fallback to np.dtype(...) for the rest
    """
    import builtins

    import numpy as np

    # Robust access to bigint_ even if defined later in the module
    _bigint_scalar = globals().get("bigint_")

    # ---- Arkouda bigint family (catch these FIRST) ----
    if (
        (isinstance(x, str) and x.lower() == "bigint")
        or isinstance(x, bigint)  # sentinel instance
        or x is bigint  # sentinel class object
        or getattr(x, "name", "").lower() == "bigint"
        or (isinstance(x, type) and x.__name__ == "bigint")  # class by name
        or (_bigint_scalar is not None and isinstance(x, _bigint_scalar))  # scalar instance
        or (isinstance(x, type) and x.__name__ == "bigint_")  # scalar class object
    ):
        return bigint()

    # ---- String dtype spellings (no Strings sentinel support here) ----
    if isinstance(x, str) and x.lower() in {"str", "str_"}:
        return np.dtype(np.str_)
    if x in (str, np.str_):
        return np.dtype(np.str_)

    # ---- Core Python scalar types ----
    if x is float:
        return np.dtype(np.float64)
    if x is bool or x is builtins.bool:
        return np.dtype(np.bool_)

    # Normalize NumPy integer scalars to Python int so they reuse the same path
    if isinstance(x, np.integer):
        x = int(x)

    # Magnitude-aware routing for Python ints
    if isinstance(x, int):
        _INT64_MIN = -(1 << 63)
        _INT64_MAX = (1 << 63) - 1
        _UINT64_MAX = (1 << 64) - 1
        if x < 0:
            # negative: fits in int64?
            return bigint() if x < _INT64_MIN else np.dtype(np.int64)
        else:
            # non-negative: prefer int64 up to max, then uint64 window, else bigint
            if x <= _INT64_MAX:
                return np.dtype(np.int64)
            if x <= _UINT64_MAX:
                return np.dtype(np.uint64)
            return bigint()

    if isinstance(x, float):
        return np.dtype(np.float64)
    if isinstance(x, bool):
        return np.dtype(np.bool_)

    # ---- Fallback to NumPy dtype for everything else ----
    try:
        return np.dtype(x)
    except TypeError as e:
        # Re-raise with a clearer message including the repr of x
        raise TypeError(f"Unsupported dtype-like object for arkouda.numpy.dtype: {x!r}") from e


_dtype_for_chapel = dict()  # type: ignore


_dtype_name_for_chapel = {  # see DType
    "real": "float64",
    "real(32)": "float32",
    "real(64)": "float64",
    "complex": "complex128",
    "complex(64)": "complex64",
    "complex(128)": "complex128",
    "int": "int64",
    "int(8)": "int8",
    "int(16)": "int16",
    "int(32)": "int32",
    "int(64)": "int64",
    "uint": "uint64",
    "uint(8)": "uint8",
    "uint(16)": "uint16",
    "uint(32)": "uint32",
    "uint(64)": "uint64",
    "bool": "bool",
    "bigint": "bigint",
    "string": "str",
}


def dtype_for_chapel(type_name: str):
    """
    Returns dtype() for the given Chapel type.

    Parameters
    ----------
    type_name : str
        The name of the Chapel type, with or without the bit width

    Returns
    -------
    dtype
        The corresponding Arkouda dtype object

    Raises
    ------
    TypeError
        Raised if Arkouda does not have a type that corresponds to `type_name`

    """
    try:
        return _dtype_for_chapel[type_name]
    except KeyError:
        try:
            dtype_name = _dtype_name_for_chapel[type_name]
        except KeyError:
            raise TypeError(f"Arkouda does not have a dtype that corresponds to '{type_name}' in Chapel")
        result = dtype(dtype_name)
        _dtype_for_chapel[type_name] = result
        return result


def can_cast(from_dt, to_dt, casting: str = "safe") -> bool:
    """
    NumPy-like can_cast with Arkouda bigint support.

    Rules (safe/default):
      • bigint → bigint                        True
      • bigint → float (any)                   True   (magnitude preserved; precision may round)
      • bigint → signed/unsigned integers      False  (possible overflow)
      • int64/uint64 → bigint                  True   (widen)
      • float → bigint                         False  (information loss)
      • otherwise                              defer to numpy.can_cast(...)
    """
    import numpy as np

    def _is_bigint_like(x) -> bool:
        if x is bigint or isinstance(x, bigint):
            return True
        if getattr(x, "name", "").lower() == "bigint":
            return True
        if isinstance(x, str) and x.lower() == "bigint":
            return True
        _bigint_scalar = globals().get("bigint_")
        if _bigint_scalar is not None and isinstance(x, _bigint_scalar):
            return True
        if isinstance(x, type) and x.__name__ in ("bigint", "bigint_"):
            return True
        return False

    def _to_np_dtype(x):
        # normalize a dtype-ish into np.dtype, but NEVER feed bigint to NumPy
        if isinstance(x, np.dtype):
            return x
        if isinstance(x, type):
            return np.dtype(x)
        if hasattr(x, "dtype") and not isinstance(x, type):
            dt = getattr(x, "dtype")
            if isinstance(dt, np.dtype):
                return dt
            return np.dtype(dt)
        return np.dtype(x)

    from_is_big = _is_bigint_like(from_dt)
    to_is_big = _is_bigint_like(to_dt)

    if from_is_big and to_is_big:
        return True

    if from_is_big:
        # bigint → NumPy dtype family
        np_to = _to_np_dtype(to_dt)
        if np.issubdtype(np_to, np.floating):
            return True
        if np_to.kind in ("i", "u"):  # integer kinds
            return False  # unsafe under "safe" policy
        if np_to.kind == "O":  # object
            return True
        # default conservative
        return False

    if to_is_big:
        # → bigint (widen from integers; floats lose info under "safe")
        try:
            np_from = _to_np_dtype(from_dt)
        except TypeError:
            # If it's bigint-like we would have returned earlier;
            # otherwise, let NumPy decide below.
            np_from = None
        if isinstance(np_from, np.dtype):
            if np.issubdtype(np_from, np.integer):
                return True
            if np.issubdtype(np_from, np.floating):
                return False
        # Non-numpy dtypes: conservative default
        return False

    # Neither side is bigint → exactly match NumPy
    return bool(np.can_cast(_to_np_dtype(from_dt), _to_np_dtype(to_dt), casting=casting))


def result_type(*args):
    """
    NumPy-like result_type with Arkouda bigint support.

    Rules:
      • If any arg is bigint-like:
          – with any float → float64
          – otherwise → ak.bigint()
      • Else defer to numpy.result_type(...)
    """
    import numpy as np

    def _is_bigint_like(x) -> bool:
        # accept sentinel instance/class, name, scalar instance/class
        if x is bigint or isinstance(x, bigint):
            return True
        if getattr(x, "name", "").lower() == "bigint":
            return True
        if isinstance(x, str) and x.lower() == "bigint":
            return True
        _bigint_scalar = globals().get("bigint_")
        if _bigint_scalar is not None and isinstance(x, _bigint_scalar):
            return True
        if isinstance(x, type) and x.__name__ in ("bigint", "bigint_"):
            return True
        return False

    has_bigint = False
    has_float = False
    np_args: list[np.dtype] = []

    for a in args:
        # 0) bigint-like → short-circuit, do not feed to NumPy
        if _is_bigint_like(a):
            has_bigint = True
            continue

        # 1) explicit numpy dtype object
        if isinstance(a, np.dtype):
            np_dt = a

        # 2) type objects (np.float64, np.int64, bool, int, etc.)
        elif isinstance(a, type):
            # bigint-like types already caught above
            np_dt = np.dtype(a)

        # 3) Python / NumPy scalar values (ints, floats, numpy numbers)
        elif isinstance(a, (int, float, bool, np.number)):
            if isinstance(a, (int, np.integer)):
                ak_dt = dtype(a)  # uses your magnitude-aware int routing
                if _is_bigint_like(ak_dt):
                    has_bigint = True
                    continue
                np_dt = np.dtype(ak_dt)
            else:
                np_dt = np.result_type(a)

        # 4) instances with a real .dtype (arrays, pdarray, numpy scalars)
        elif hasattr(a, "dtype") and not isinstance(a, type):
            dt = getattr(a, "dtype")
            if _is_bigint_like(dt):
                has_bigint = True
                continue
            np_dt = np.dtype(dt)

        # 5) generic fallback
        else:
            np_dt = np.result_type(a)

        np_dt = np.dtype(np_dt)
        np_args.append(np_dt)
        if np.issubdtype(np_dt, np.floating):
            has_float = True

    if has_bigint:
        return np.dtype(np.float64) if has_float else bigint()

    return np.result_type(*np_args)


def _is_dtype_in_union(dtype, union_type) -> builtins.bool:
    """
    Check if a given type is in a typing.Union.

    Args
    ----
        dtype (type): The type to check for.
        union_type (type): The typing.Union type to check against.

    Returns
    -------
        bool True if the dtype is in the union_type, False otherwise.
    """
    return hasattr(union_type, "__args__") and dtype in union_type.__args__


def _val_isinstance_of_union(val, union_type) -> builtins.bool:
    """
    Check if a given val is an instance of one of the types in the typing.Union.

    Args
    ----
        val: The val to do the isinstance check on.
        union_type (type): The typing.Union type to check against.

    Returns
    -------
        bool: True if the val is an instance of one
            of the types in the union_type, False otherwise.
    """
    return hasattr(union_type, "__args__") and isinstance(val, union_type.__args__)


# --- new code (near your bigint definition) ---


class _BigIntMeta(type):
    def __call__(cls, *args, **kwargs):
        # ak.bigint()  -> dtype sentinel (preserve current behavior)
        if not args and not kwargs:
            return super().__call__()
        # ak.bigint(1) -> scalar, like np.int64(1)
        return bigint_(args[0] if args else 0)


class bigint(metaclass=_BigIntMeta):
    """
    Arkouda dtype sentinel for variable-width integers.

    - ak.bigint() returns the dtype sentinel (singleton)
    - ak.bigint(…) returns a bigint_ scalar (see below)
    """

    __slots__ = ()
    name: str = "bigint"
    kind: str = "ui"
    itemsize: int = 128
    ndim: int = 0
    shape: tuple = ()
    _INSTANCE = None

    def __new__(cls):
        inst = getattr(cls, "_INSTANCE", None)
        if inst is None:
            inst = super().__new__(cls)
            cls._INSTANCE = inst
        return inst

    def __reduce__(self):
        return (bigint, ())

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"dtype({self.name})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if other is self or other is bigint:
            return True
        if isinstance(other, str):
            return other.lower() == "bigint"
        name = getattr(other, "name", None)
        return name == "bigint"

    def __ne__(self, other):
        return not self.__eq__(other)

    # dtype-like conversion hook (kept for compatibility)
    def type(self, x):
        return int(x)

    @property
    def is_signed(self) -> bool:
        return True

    @property
    def is_variable_width(self) -> bool:
        return True


intTypes = frozenset((dtype("int64"), dtype("uint64"), dtype("uint8")))
bitType = uint64

# Union aliases used for static and runtime type checking
bool_scalars = Union[builtins.bool, np.bool_]
float_scalars = Union[float, np.float64, np.float32]
int_scalars = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
numeric_scalars = Union[float_scalars, int_scalars]
numeric_and_bool_scalars = Union[bool_scalars, numeric_scalars]
numpy_scalars = Union[
    np.float64,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.bool_,
    np.str_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
str_scalars = Union[str, np.str_]
all_scalars = Union[bool_scalars, numeric_scalars, numpy_scalars, str_scalars]

"""
The DType enum defines the supported Arkouda data types in string form.
"""


class DType(Enum):
    FLOAT = "float"
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    INT = "int"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT = "uint"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOL = "bool"
    BIGINT = "bigint"
    STR = "str"

    def __str__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter.
        """
        return self.value

    def __repr__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter.
        """
        return self.value


# --- bigint scalar type ---


class bigint_(int):
    """Arkouda bigint scalar (inherits Python int for arbitrary precision)."""

    __slots__ = ()

    def __new__(cls, x=0):
        if isinstance(x, (str, bytes)):
            val = int(x, 0)
        else:
            val = int(x)
        return super().__new__(cls, val)

    @property
    def dtype(self):
        return bigint()

    def item(self):
        return int(self)

    def __repr__(self):
        return f"ak.bigint_({int(self)})"


ARKOUDA_SUPPORTED_BOOLS = (builtins.bool, np.bool_)


ARKOUDA_SUPPORTED_INTS = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    bigint,
    bigint_,
)

ARKOUDA_SUPPORTED_FLOATS = (float, np.float64, np.float32)
ARKOUDA_SUPPORTED_NUMBERS = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    bigint,
    bigint_,
)

# TODO: bring supported data types into parity with all numpy dtypes
# missing full support for: float32, int32, int16, int8, uint32, uint16, complex64, complex128
# ARKOUDA_SUPPORTED_DTYPES = frozenset([member.value for _, member in DType.__members__.items()])
ARKOUDA_SUPPORTED_DTYPES = (
    bool_,
    float,
    float64,
    int,
    int64,
    uint64,
    uint8,
    bigint,
    str,
)

DTypes = frozenset([member.value for _, member in DType.__members__.items()])
DTypeObjects = frozenset([bool_, float, float64, int, int64, str, str_, uint8, uint64])
NumericDTypes = frozenset(["bool_", "bool", "float", "float64", "int", "int64", "uint64", "bigint"])
SeriesDTypes = {
    "string": np.str_,
    "<class 'str'>": np.str_,
    "int64": np.int64,
    "uint64": np.uint64,
    "<class 'numpy.int64'>": np.int64,
    "float64": np.float64,
    "<class 'numpy.float64'>": np.float64,
    "bool": np.bool_,
    "<class 'bool'>": np.bool_,
    "datetime64[ns]": np.int64,
    "timedelta64[ns]": np.int64,
}
ScalarDTypes = frozenset(["bool_", "float64", "int64"])


def isSupportedInt(num):
    """
    Whether a scalar is an arkouda supported integer dtype.

    Parameters
    ----------
    num: object
        A scalar.

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported integer dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedInt(79)
    True
    >>> ak.isSupportedInt(54.9)
    False

    """
    return isinstance(num, ARKOUDA_SUPPORTED_INTS)


def isSupportedFloat(num):
    """
    Whether a scalar is an arkouda supported float dtype.

    Parameters
    ----------
    num: object
        A scalar.

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported float dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedFloat(56)
    False
    >>> ak.isSupportedFloat(56.7)
    True

    """
    return isinstance(num, ARKOUDA_SUPPORTED_FLOATS)


def isSupportedNumber(num):
    """
    Whether a scalar is an arkouda supported numeric dtype.

    Parameters
    ----------
    num: object
        A scalar.

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported numeric dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedNumber(45.9)
    True
    >>> ak.isSupportedNumber("string")
    False

    """
    return isinstance(num, ARKOUDA_SUPPORTED_NUMBERS)


def isSupportedBool(num):
    """
    Whether a scalar is an arkouda supported boolean dtype.

    Parameters
    ----------
    num: object
        A scalar.

    Returns
    -------
    bool
        True if scalar is an instance of an arkouda supported boolean dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedBool("True")
    False
    >>> ak.isSupportedBool(True)
    True

    """
    return isinstance(num, ARKOUDA_SUPPORTED_BOOLS)


def isSupportedDType(scalar: object) -> builtins.bool:
    """
    Whether a scalar is an arkouda supported dtype.

    Parameters
    ----------
    scalar: object

    Returns
    -------
    builtins.bool
        True if scalar is an instance of an arkouda supported dtype, else False.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.isSupportedDType(ak.int64(64))
    True
    >>> ak.isSupportedDType(np.complex128(1+2j))
    False

    """
    return isinstance(scalar, ARKOUDA_SUPPORTED_DTYPES)


def resolve_scalar_dtype(val: object) -> str:
    """
    Try to infer what dtype arkouda_server should treat val as.

    Parameters
    ----------
    val: object
        The object to determine the dtype of.

    Return
    ------
    str
        The dtype name, if it can be resolved, otherwise the type (as str).

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.resolve_scalar_dtype(1)
    'int64'
    >>> ak.resolve_scalar_dtype(2.0)
    'float64'

    """
    # Python builtins.bool or np.bool
    if isinstance(val, builtins.bool) or (
        hasattr(val, "dtype") and cast(np.bool_, val).dtype.kind == "b"
    ):
        return "bool"
    # Python int or np.int* or np.uint*
    elif isinstance(val, int) or (hasattr(val, "dtype") and cast(np.uint, val).dtype.kind in "ui"):
        # we've established these are int, uint, or bigint,
        # so we can do comparisons
        if isSupportedInt(val) and val >= 2**64:  # type: ignore
            return "bigint"
        elif isinstance(val, np.uint64) or val >= 2**63:  # type: ignore
            return "uint64"
        else:
            return "int64"
    # Python float or np.float*
    elif isinstance(val, float) or (hasattr(val, "dtype") and cast(np.float64, val).dtype.kind == "f"):
        return "float64"
    elif isinstance(val, complex) or (hasattr(val, "dtype") and cast(np.float64, val).dtype.kind == "c"):
        return "float64"  # TODO: actually support complex values in the backend
    elif isinstance(val, builtins.str) or isinstance(val, np.str_):
        return "str"
    # Other numpy dtype
    elif hasattr(val, "dtype"):
        return cast(np.dtype, val).name
    # Other python type
    else:
        return builtins.str(type(val))


def get_byteorder(dt: np.dtype) -> str:
    """
    Get a concrete byteorder (turns '=' into '<' or '>') on the client.

    Parameters
    ----------
    dt: np.dtype
        The numpy dtype to determine the byteorder of.

    Return
    ------
    str
        Returns "<" for little endian and ">" for big endian.

    Raises
    ------
    ValueError
        Returned if sys.byteorder is not "little" or "big"

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.get_byteorder(ak.dtype(ak.int64))
    '<'

    """
    if dt.byteorder == "=":
        if sys.byteorder == "little":
            return "<"
        elif sys.byteorder == "big":
            return ">"
        else:
            raise ValueError("Client byteorder must be 'little' or 'big'")
    else:
        return dt.byteorder


def get_server_byteorder() -> str:
    """
    Get the server's byteorder.

    Return
    ------
    str
        Returns "little" for little endian and "big" for big endian.

    Raises
    ------
    ValueError
        Raised if Server byteorder is not 'little' or 'big'

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.get_server_byteorder()
    'little'

    """
    from arkouda.client import get_config

    order = get_config()["byteorder"]
    if order not in ("little", "big"):
        raise ValueError("Server byteorder must be 'little' or 'big'")
    return cast("str", order)
