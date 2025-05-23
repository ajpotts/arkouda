from __future__ import annotations

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
