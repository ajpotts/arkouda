from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.strings import Strings
from arkouda.pandas.categorical import Categorical

AkArray = Union[pdarray, Strings, Categorical]


def _factorize_ak_level(a: AkArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (codes, uniques_np) for one Arkouda array level.
    Uses the fastest available Arkouda path; falls back to NumPy if needed.
    """
    # Fast paths if your Arkouda build exposes factorization-like helpers.
    # Prefer these if available in your codebase:
    #
    # 1) If ak has factorize-like API:
    # try:
    #     codes_ak, uniques_ak = ak.factorize(a)  # hypothetical
    #     return codes_ak.to_ndarray(), uniques_ak.to_ndarray()
    # except Exception:
    #     pass
    #
    # 2) ak.unique with inverse (if your version supports it):
    # try:
    #     uniques_ak, inverse = ak.unique(a, return_inverse=True)
    #     return inverse.to_ndarray(), uniques_ak.to_ndarray()
    # except Exception:
    #     pass

    # Conservative, always-works fallback:
    if hasattr(a, "to_ndarray"):
        vals = a.to_ndarray()
    elif hasattr(a, "to_list"):  # e.g., some ak.Strings versions
        vals = np.asarray(a.to_list(), dtype=object)
    else:
        # Last resort—force materialization via Python iter
        vals = np.asarray(list(a), dtype=object)

    codes, uniques = pd.factorize(vals, sort=False)  # codes: np.intp, uniques: np.ndarray
    return codes, uniques


class ArkoudaMultiIndex(pd.MultiIndex):
    """
    A convenience subclass that *constructs* a pandas.MultiIndex from Arkouda
    level arrays. Internally it’s still a pandas.MultiIndex (NumPy codes),
    but you get nice constructors that accept ak arrays and helper roundtrips.
    """

    # ---- Preferred constructors ------------------------------------------------

    @classmethod
    def from_ak_arrays(
        cls,
        levels: Sequence[AkArray],
        names: Sequence[str] | None = None,
    ) -> "ArkoudaMultiIndex":
        """
        Build a MultiIndex from a list/tuple of Arkouda arrays, one per level.
        Each level is factorized independently.
        """
        np_levels: list[pd.Index] = []
        np_codes: list[np.ndarray] = []

        for a in levels:
            codes, uniques = _factorize_ak_level(a)
            np_levels.append(pd.Index(uniques))
            np_codes.append(codes.astype(np.intp, copy=False))

        mi = pd.MultiIndex(levels=np_levels, codes=np_codes, names=names, verify_integrity=False)
        # Recast as our subclass (keeps all pandas internals intact)
        mi.__class__ = cls
        return mi

    @classmethod
    def from_ak_tuples(
        cls,
        arrays: Sequence[AkArray],
        names: Sequence[str] | None = None,
    ) -> "ArkoudaMultiIndex":
        """
        Equivalent to pd.MultiIndex.from_arrays on the *materialized* tuples,
        but factorizes level-by-level without first forming Python tuples.
        """
        return cls.from_ak_arrays(arrays, names=names)

    @classmethod
    def from_ak_product(
        cls,
        levels: Sequence[AkArray],
        names: Sequence[str] | None = None,
    ) -> "ArkoudaMultiIndex":
        """
        Cartesian product of Arkouda levels (like pd.MultiIndex.from_product).
        Implemented by materializing each level’s uniques, then delegating.
        """
        uniq_levels = []
        for a in levels:
            _, uniq = _factorize_ak_level(a)
            uniq_levels.append(pd.Index(uniq))
        mi = pd.MultiIndex.from_product(uniq_levels, names=names)
        mi.__class__ = cls
        return mi

    # ---- Convenience roundtrips ------------------------------------------------

    def to_ak_codes(self) -> Tuple[pdarray, ...]:
        """Return the codes for each level as Arkouda int arrays."""
        return tuple(ak_array(c.astype(np.int64, copy=False)) for c in self.codes)

    def to_ak_levels(self) -> Tuple[AkArray, ...]:
        """
        Return Arkouda arrays for each level’s unique values.
        For strings, we build ak.Strings; for other dtypes we use ak.array.
        """
        out: list[AkArray] = []
        for lev in self.levels:
            # Heuristic: strings vs numeric
            if lev.dtype == object or pd.api.types.is_string_dtype(lev.dtype):
                out.append(ak_array(lev.astype(str).to_numpy(copy=False)))
            else:
                out.append(ak_array(lev.to_numpy(copy=False)))
        return tuple(out)

    # ---- Nice constructors for common shapes ----------------------------------

    @classmethod
    def from_ak_frame(
        cls,
        cols: "dict[str, AkArray] | Sequence[Tuple[str, AkArray]]",
        names: Sequence[str] | None = None,
    ) -> "ArkoudaMultiIndex":
        """
        Build from a mapping or sequence of (name, ak_array). Orders by input.
        """
        if isinstance(cols, dict):
            keys = list(cols.keys())
            arrays = [cols[k] for k in keys]
            if names is None:
                names = keys
        else:
            keys = [k for (k, _) in cols]
            arrays = [a for (_, a) in cols]
            if names is None:
                names = keys
        return cls.from_ak_arrays(arrays, names=names)
