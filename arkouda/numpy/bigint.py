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

class bigint_(int):
    """
    Arkouda-aware integer scalar carrying an optional `max_bits` hint via side-car storage.

    - Integer ops with ints/bigint_ → bigint_ (metadata merged).
    - True division `/` → float (Python-consistent).
    - No automatic wrap/truncation by `max_bits` (pure metadata for now).
    """

    # NOTE: no __slots__ — not supported for int subclasses

    def __new__(cls, value: Union[int, "bigint_"], max_bits: Optional[int] = None):
        iv = int(value)
        obj = super().__new__(cls, iv)
        # inherit from rhs if not explicitly provided
        if max_bits is None and isinstance(value, bigint_):
            max_bits = value.max_bits
        _set_max_bits(obj, max_bits)
        return obj

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

    # -------- repr --------
    def __repr__(self) -> str:
        mb = f", max_bits={self.max_bits}" if self.max_bits is not None else ""
        return f"bigint_({int(self)}{mb})"
