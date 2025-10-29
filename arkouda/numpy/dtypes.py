from __future__ import annotations

import builtins
from enum import Enum
import sys
from typing import TYPE_CHECKING, List, Union, cast
from .bigint import bigint_, bigint
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


if TYPE_CHECKING:
    from arkouda.pdarrayclass import pdarray

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

_U64_THRESHOLD = 1 << 64  # 2**64


def dtype(dtype):
    """
    Create a data type object.

    Parameters
    ----------
    dtype: object
        Object to be converted to a data type object.

    Returns
    -------
    type

    """
    # we had to create our own bigint type since numpy
    # gives them dtype=object there's no np equivalent
    if (
        (isinstance(dtype, str) and dtype.lower() == "bigint")
        or isinstance(dtype, bigint)
        or (hasattr(dtype, "name") and dtype.name == "bigint")

        or dtype is bigint_ or isinstance(dtype, bigint_)
    ):
        return bigint()
    if isinstance(dtype, str) and dtype in ["Strings"]:
        return np.dtype(np.str_)

    # Bool must come first since bool is a subclass of int
    if isinstance(dtype, (bool, np.bool_)):
        return np.dtype(np.bool_)

    # Python / NumPy integer scalar -> route big magnitudes to ak.bigint()
    if isinstance(dtype, (int, np.integer)):
        # Normalize to Python int for magnitude check
        iv = int(dtype)
        if abs(iv) >= _U64_THRESHOLD:
            return bigint()
        return np.dtype(np.int64)

    if isinstance(dtype, int):
        if 0 < dtype and dtype < 2**64:
            return np.dtype(np.uint64)
        if dtype >= 2**64:
            return bigint()
        else:
            return np.dtype(np.int64)
    if isinstance(dtype, float):
        return np.dtype(np.float64)
    if isinstance(dtype, builtins.bool):
        return np.dtype(np.bool)
    return np.dtype(dtype)


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

    Rules
    -----
    • bigint → bigint          : True
    • bigint → float64         : True   (magnitude-preserving)
    • bigint → int64           : False  (avoid truncation)
    • int/uint/bool → bigint   : True   (widening)
    • otherwise                : defer to NumPy
    """
    import numpy as np
    from .bigint import bigint as AKBigint  # dtype class

    # ---- recognizers that NEVER call np.dtype on bigint-y things ----
    def _is_bigint_obj(x) -> bool:
        if x is AKBigint or isinstance(x, AKBigint):
            return True
        if isinstance(x, str) and x.lower() == "bigint":
            return True
        try:
            if isinstance(x, np.dtype) and x.name == "bigint":
                return True
        except Exception:
            pass
        return getattr(x, "name", None) == "bigint"

    # robustly map non-bigint objects (including Python/NumPy scalars) -> np.dtype
    def _norm_non_bigint(x) -> np.dtype:
        if isinstance(x, np.dtype):
            return x
        # Python/NumPy scalars → dtype via result_type
        if isinstance(x, (bool, int, float, complex, np.generic)):
            return np.result_type(x)
        # numpy arrays and objects with .dtype
        if hasattr(x, "dtype"):
            return np.dtype(getattr(x, "dtype"))
        # string dtype tokens (e.g., "int64", "float64")
        if isinstance(x, str):
            return np.dtype(x)
        # last resort
        return np.dtype(x)

    def _is_dt(x, target: np.dtype) -> bool:
        return isinstance(x, np.dtype) and x == target

    fb = _is_bigint_obj(from_dt)
    tb = _is_bigint_obj(to_dt)

    # ---- explicit bigint rules ----
    if fb and tb:
        return True

    if fb and not tb:
        t = _norm_non_bigint(to_dt)
        if _is_dt(t, np.dtype(np.float64)):
            return True
        if _is_dt(t, np.dtype(np.int64)):
            return False
        # default for other non-bigint targets
        return np.can_cast(np.dtype("object"), t, casting=casting)

    if tb and not fb:
        f = _norm_non_bigint(from_dt)
        if f.kind in ("i", "u", "b"):  # widening to bigint
            return True
        if f.kind == "f":              # float → bigint is lossy under "safe"
            return False
        return np.can_cast(f, np.dtype("object"), casting=casting)

    # ---- default: no bigint involved ----
    f = _norm_non_bigint(from_dt)
    t = _norm_non_bigint(to_dt)
    return np.can_cast(f, t, casting=casting)



def result_type(*args):
    """
    NumPy-like result_type with Arkouda bigint support.

    Returns:
      - np.dtype(...) for NumPy dtypes
      - ak.bigint() (the dtype *instance*, not the class) when bigint wins
    """
    import numpy as np
    from .dtypes import dtype as ak_dtype
    from .bigint import bigint as ak_bigint  # dtype-class; call it to get instance

    def _is_bigint_like(x) -> bool:
        # accept dtype instance, dtype class, name, or arrays/scalars with bigint dtype
        if x is ak_bigint:
            return True
        if x is ak_bigint():
            return True
        if isinstance(x, str) and x.lower() == "bigint":
            return True
        try:
            # np.dtype(...) will resolve ak_bigint() correctly
            dt = np.dtype(x)
            return getattr(dt, "name", "") == "bigint"
        except Exception:
            pass
        # objects with .dtype
        if hasattr(x, "dtype"):
            try:
                return getattr(np.dtype(getattr(x, "dtype")), "name", "") == "bigint"
            except Exception:
                return False
        return False

    has_bigint = False
    has_float = False
    np_args = []

    saw_unsigned = False
    signed_from_nonneg_scalar = False
    all_integer = True

    for a in args:
        # normalize explicit bigint signals up front
        if _is_bigint_like(a):
            has_bigint = True
            continue

        # accept numpy dtype / type / scalars / arrays
        if isinstance(a, np.dtype):
            np_dt = a
        elif isinstance(a, type):
            np_dt = np.dtype(a)
        elif hasattr(a, "dtype") and not isinstance(a, type):
            # arrays / numpy scalars
            adt = getattr(a, "dtype")
            if _is_bigint_like(adt):
                has_bigint = True
                continue
            np_dt = np.dtype(adt)
        elif isinstance(a, (bool, np.bool_)):
            np_dt = np.dtype(np.bool_)
        elif isinstance(a, (int, np.integer)):
            # route large magnitude ints to bigint via ak.dtype (your scalar logic)
            dt = ak_dtype(a)
            if _is_bigint_like(dt):
                has_bigint = True
                continue
            np_dt = np.dtype(dt)
            if np_dt.kind == "i" and int(a) >= 0:
                signed_from_nonneg_scalar = True
        elif isinstance(a, (float, np.floating)):
            np_dt = np.result_type(a)
        else:
            # generic fallback
            try:
                np_dt = np.result_type(a)
            except Exception:
                # if NumPy can’t interpret it and it’s not bigint-like, rethrow
                raise

        np_dt = np.dtype(np_dt)
        np_args.append(np_dt)

        if not np.issubdtype(np_dt, np.integer):
            all_integer = False
        if np_dt.kind == "u":
            saw_unsigned = True
        if np.issubdtype(np_dt, np.floating):
            has_float = True

    # ---- bigint promotion rules ----
    if has_bigint:
        return np.dtype(np.float64) if has_float else ak_bigint()  # << return INSTANCE

    # ---- unsigned × non-neg signed scalar: keep unsigned width ----
    if all_integer and saw_unsigned and signed_from_nonneg_scalar:
        unsigneds = [dt for dt in np_args if dt.kind == "u"]
        return max(unsigneds, key=lambda d: d.itemsize)

    # ---- default to NumPy ----
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

    Rules (scalar-ish inputs):
    - bool / np.bool_                    -> "bool"
    - ak.bigint_ (custom scalar)         -> "bigint"
    - Python int / np.integer:
        * if abs(int(val)) >= 2**64      -> "bigint"
        * elif unsigned OR int(val) >= 2**63 -> "uint64"
        * else                           -> "int64"
    - float / np.floating                -> "float64"
    - complex / np.complexfloating       -> "float64"  (backend TODO for complex)
    - str / np.str_                      -> "str"
    - anything with a dtype              -> dtype.name (best-effort)
    - otherwise                          -> str(type(val))
    """
    import builtins
    import numpy as np
    from .bigint import bigint_  # your scalar class

    U64 = 1 << 64
    I63 = 1 << 63  # threshold to decide uint64 vs int64 for plain ints

    # ---- bool first (since bool is a subclass of int) ----
    if isinstance(val, (builtins.bool, np.bool_)):
        return "bool"

    # ---- your custom bigint scalar ----
    if isinstance(val, bigint_):
        return "bigint"

    # ---- Python / NumPy integer scalars ----
    if isinstance(val, (int, np.integer)):
        iv = int(val)  # normalize numpy integers to Python int
        if abs(iv) >= U64:
            return "bigint"
        # If it's an unsigned numpy integer (any width), treat as uint64
        if isinstance(val, np.unsignedinteger) or iv >= I63:
            return "uint64"
        return "int64"

    # ---- floats ----
    if isinstance(val, (float, np.floating)):
        return "float64"

    # ---- complex (map to float64 until backend support exists) ----
    if isinstance(val, (complex, np.complexfloating)):
        return "float64"

    # ---- strings ----
    if isinstance(val, (builtins.str, np.str_)):
        return "str"

    # ---- things that carry a dtype (e.g., numpy scalar/array) ----
    if hasattr(val, "dtype"):
        try:
            return np.dtype(getattr(val, "dtype")).name
        except Exception:
            pass  # fall through to generic

    # ---- generic fallback ----
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
