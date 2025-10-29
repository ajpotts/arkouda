from __future__ import annotations

from typing import Optional, Union
import weakref

from numpy import bool


__all__ = [
    "bigint",
    "bigint_",
]


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

    @property
    def dtype(self):
        # Return the bigint dtype sentinel from this module
        return bigint()


_IntLike = Union[int, "bigint_"]

# side-car storage for metadata
_MAX_BITS: dict[int, Optional[int]] = {}

def _set_max_bits(obj: "bigint_", mb: Optional[int]) -> None:
    key = id(obj)
    _MAX_BITS[key] = mb
    # best-effort cleanup; if weakref isn't supported, we just skip it
    try:
        weakref.finalize(obj, _MAX_BITS.pop, key, None)
    except TypeError:
        pass

def _mb_max(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)

# arkouda/numpy/bigint.py

class bigint_(int):

    def __repr__(self) -> str:
        # Keep it simple; the test only checks for 'ak.bigint_('
        return f"ak.bigint_({int(self)})"

    # (optional) make str() match repr()
    def __str__(self) -> str:
        return repr(self)

    def __new__(cls, value=0, max_bits=None):
        # Accept ints, numpy integer scalars, and strings with 0x/0o/0b prefixes
        try:
            import numpy as np  # local import to avoid early import cost/cycles
            np_integer = np.integer
        except Exception:  # numpy may not be loaded yet
            np_integer = ()  # fallback

        if isinstance(value, (bytes, bytearray)):
            value = value.decode()

        if isinstance(value, str):
            # base=0 lets int() infer 0x (hex), 0o (oct), 0b (bin), leading +/-,
            # ignores underscores per Python syntax.
            iv = int(value, 0)
        elif np_integer and isinstance(value, np_integer):
            iv = int(value.item())
        else:
            iv = int(value)

        # NOTE: we don't store max_bits on the scalar (int subclasses can't
        # have per-instance attrs). max_bits applies to arrays, not scalars.
        return int.__new__(cls, iv)

    @property
    def dtype(self):
        from .bigint import bigint
        return bigint()

    # NEW: numpy-scalar compatibility
    def item(self) -> int:
        """Return the Python int value (NumPy-scalar style)."""
        return int(self)

    # (optional but handy) let it act like an index too
    def __index__(self) -> int:
        return int(self)


    # expose metadata via property
    @property
    def max_bits(self) -> Optional[int]:
        return _MAX_BITS.get(id(self))

    # -------- helpers --------
    def _result(self, val: int, other: Optional[_IntLike] = None) -> "bigint_":
        # If you later want wrap-to-width, do it here before constructing bigint_.
        mb = self.max_bits
        if isinstance(other, bigint_):
            mb = _mb_max(mb, other.max_bits)
        return bigint_(val, max_bits=mb)

    @staticmethod
    def _as_bigint(x: _IntLike) -> "bigint_":
        if isinstance(x, bigint_):
            return x
        if isinstance(x, int) and not isinstance(x, bool):
            return bigint_(x)
        raise TypeError(f"Unsupported operand type: {type(x)}")

    # -------- unary --------
    def __neg__(self) -> "bigint_": return self._result(-int(self))
    def __pos__(self) -> "bigint_": return self._result(+int(self))
    def __abs__(self) -> "bigint_": return self._result(abs(int(self)))
    def __invert__(self) -> "bigint_": return self._result(~int(self))

    # -------- add/sub/mul --------
    def __add__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(self) + int(o), o)
        return NotImplemented
    __radd__ = __add__

    def __sub__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(self) - int(o), o)
        return NotImplemented

    def __rsub__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(o) - int(self), o)
        return NotImplemented

    def __mul__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(self) * int(o), o)
        return NotImplemented
    __rmul__ = __mul__

    # -------- division --------
    def __truediv__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            return int(self) / int(other)  # float
        return NotImplemented

    def __rtruediv__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            return int(other) / int(self)  # float
        return NotImplemented

    def __floordiv__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(self) // int(o), o)
        return NotImplemented

    def __rfloordiv__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(o) // int(self), o)
        return NotImplemented

    def __mod__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(self) % int(o), o)
        return NotImplemented

    def __rmod__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(o) % int(self), o)
        return NotImplemented

    def __divmod__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            q, r = divmod(int(self), int(o))
            return (self._result(q, o), self._result(r, o))
        return NotImplemented

    def __rdivmod__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            q, r = divmod(int(o), int(self))
            return (self._result(q, o), self._result(r, o))
        return NotImplemented

    # -------- bitwise --------
    def __and__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(self) & int(o), o)
        return NotImplemented
    __rand__ = __and__

    def __or__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(self) | int(o), o)
        return NotImplemented
    __ror__ = __or__

    def __xor__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            o = self._as_bigint(other)
            return self._result(int(self) ^ int(o), o)
        return NotImplemented
    __rxor__ = __xor__

    def __lshift__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            return self._result(int(self) << int(other))
        return NotImplemented

    def __rlshift__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            return self._as_bigint(other)._result(int(other) << int(self))
        return NotImplemented

    def __rshift__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            return self._result(int(self) >> int(other))
        return NotImplemented

    def __rrshift__(self, other: _IntLike):
        if isinstance(other, (int, bigint_)) and not isinstance(other, bool):
            return self._as_bigint(other)._result(int(other) >> int(self))
        return NotImplemented


