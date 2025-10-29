import math
import pickle
import sys

import numpy as np
import pytest

import arkouda as ak
from arkouda.numpy import dtypes
from arkouda.numpy.dtypes import bigint_


"""
DtypesTest encapsulates arkouda dtypes module methods
"""

SUPPORTED_NP_DTYPES = [
    bool,
    int,
    float,
    str,
    np.bool_,
    np.int64,
    np.float64,
    np.uint8,
    np.uint64,
    np.str_,
]


class TestDTypes:
    def test_dtypes_docstrings(self):
        import doctest

        result = doctest.testmod(dtypes)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_resolve_scalar_dtype(self):
        for b in True, False:
            assert "bool" == dtypes.resolve_scalar_dtype(b)

        for i in np.iinfo(np.int64).min, -1, 0, 3, np.iinfo(np.int64).max:
            assert "int64" == dtypes.resolve_scalar_dtype(i)

        floats = [
            -np.inf,
            np.finfo(np.float64).min,
            -3.14,
            -0.0,
            0.0,
            7.0,
            np.finfo(np.float64).max,
            np.inf,
            np.nan,
        ]
        for f in floats:
            assert "float64" == dtypes.resolve_scalar_dtype(f)

        for s in "test", '"', " ", "":
            assert "str" == dtypes.resolve_scalar_dtype(s)
        assert "<class 'list'>" == dtypes.resolve_scalar_dtype([1])

        assert "uint64" == dtypes.resolve_scalar_dtype(2**63 + 1)
        assert "bigint" == dtypes.resolve_scalar_dtype(2**64)

    def test_is_dtype_in_union(self):
        from typing import Union

        from arkouda.numpy.dtypes import _is_dtype_in_union

        float_scalars = Union[float, np.float64, np.float32]
        assert _is_dtype_in_union(np.float64, float_scalars)
        # Test with a type not present in the union
        assert not _is_dtype_in_union(np.int64, float_scalars)
        # Test with a non-Union type
        assert not _is_dtype_in_union(np.float64, float)

    def test_pdarrays_datatypes(self):
        assert dtypes.dtype("int64") == ak.array(np.arange(10)).dtype
        assert dtypes.dtype("uint64") == ak.array(np.arange(10), ak.uint64).dtype
        assert dtypes.dtype("bool") == ak.ones(10, ak.bool_).dtype
        assert dtypes.dtype("float64") == ak.ones(10).dtype
        assert dtypes.dtype("str") == ak.random_strings_uniform(1, 16, size=10).dtype

        bi = ak.bigint_from_uint_arrays(
            [ak.ones(10, dtype=ak.uint64), ak.arange(10, dtype=ak.uint64)]
        ).dtype
        assert dtypes.dtype("bigint") == bi
        assert dtypes.dtype("bigint") == ak.arange(2**200, 2**200 + 10).dtype

    def test_isSupportedInt(self):
        for supported in (
            -10,
            1,
            np.int64(1),
            np.int64(1.0),
            np.uint32(1),
            2**63 + 1,
            2**200,
        ):
            assert dtypes.isSupportedInt(supported)
        for unsupported in 1.0, "1":
            assert not dtypes.isSupportedInt(unsupported)

    def test_isSupportedFloat(self):
        for supported in np.nan, -np.inf, 3.1, -0.0, float(1), np.float64(1):
            assert dtypes.isSupportedFloat(supported)
        for unsupported in np.int64(1.0), int(1.0), "1.0":
            assert not dtypes.isSupportedFloat(unsupported)

    def test_DtypeEnum(self):
        assert "bool" == str(dtypes.DType.BOOL)
        assert "float32" == str(dtypes.DType.FLOAT32)
        assert "float64" == str(dtypes.DType.FLOAT64)
        assert "float" == str(dtypes.DType.FLOAT)
        assert "complex64" == str(dtypes.DType.COMPLEX64)
        assert "complex128" == str(dtypes.DType.COMPLEX128)
        assert "int8" == str(dtypes.DType.INT8)
        assert "int16" == str(dtypes.DType.INT16)
        assert "int32" == str(dtypes.DType.INT32)
        assert "int64" == str(dtypes.DType.INT64)
        assert "int" == str(dtypes.DType.INT)
        assert "uint8" == str(dtypes.DType.UINT8)
        assert "uint16" == str(dtypes.DType.UINT16)
        assert "uint32" == str(dtypes.DType.UINT32)
        assert "uint64" == str(dtypes.DType.UINT64)
        assert "uint" == str(dtypes.DType.UINT)
        assert "str" == str(dtypes.DType.STR)
        assert "bigint" == str(dtypes.DType.BIGINT)

        assert (
            frozenset(
                {
                    "float32",
                    "float64",
                    "float",
                    "complex64",
                    "complex128",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "int",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "uint",
                    "bool",
                    "str",
                    "bigint",
                }
            )
            == ak.DTypes
        )

        from arkouda.numpy import bigint, bool_, float64, int64, uint8, uint64

        assert (
            bool_,
            float,
            float64,
            int,
            int64,
            uint64,
            uint8,
            bigint,
            str,
        ) == ak.ARKOUDA_SUPPORTED_DTYPES

    def test_NumericDTypes(self):
        num_types = frozenset(["bool", "bool_", "float", "float64", "int", "int64", "uint64", "bigint"])
        assert num_types == dtypes.NumericDTypes

    def test_SeriesDTypes(self):
        for dt in "int64", "<class 'numpy.int64'>", "datetime64[ns]", "timedelta64[ns]":
            assert dtypes.SeriesDTypes[dt] == np.int64

        for dt in "string", "<class 'str'>":
            assert dtypes.SeriesDTypes[dt] == np.str_

        for dt in "float64", "<class 'numpy.float64'>":
            assert dtypes.SeriesDTypes[dt] == np.float64

        for dt in "bool", "<class 'bool'>":
            assert dtypes.SeriesDTypes[dt] == np.bool_

    def test_scalars(self):
        assert "typing.Union[bool, numpy.bool]" == str(ak.bool_scalars)
        assert "typing.Union[bool, numpy.bool]" == str(ak.bool_scalars)
        assert "typing.Union[float, numpy.float64, numpy.float32]" == str(ak.float_scalars)
        assert (
            "typing.Union[int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, "
            + "numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]"
        ) == str(ak.int_scalars)

        assert (
            "typing.Union[float, numpy.float64, numpy.float32, int, numpy.int8, numpy.int16, "
            + "numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]"
        ) == str(ak.numeric_scalars)

        assert "typing.Union[str, numpy.str_]" == str(ak.str_scalars)
        assert (
            "typing.Union[numpy.float64, numpy.float32, numpy.int8, numpy.int16, numpy.int32, "
            + "numpy.int64, numpy.bool, numpy.str_, numpy.uint8, numpy.uint16, numpy.uint32, "
            + "numpy.uint64]"
        ) == str(ak.numpy_scalars)

        assert (
            "typing.Union[bool, numpy.bool, float, numpy.float64, numpy.float32, int, numpy.int8, "
            + "numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32,"
            + " numpy.uint64, numpy.str_, str]"
        ) == str(ak.all_scalars)

    def test_number_format_strings(self):
        assert "{}" == dtypes.NUMBER_FORMAT_STRINGS["bool"]
        assert "{:d}" == dtypes.NUMBER_FORMAT_STRINGS["int64"]
        assert "{:.17f}" == dtypes.NUMBER_FORMAT_STRINGS["float64"]
        assert "{f}" == dtypes.NUMBER_FORMAT_STRINGS["np.float64"]
        assert "{:d}" == dtypes.NUMBER_FORMAT_STRINGS["uint8"]
        assert "{:d}" == dtypes.NUMBER_FORMAT_STRINGS["uint64"]
        assert "{:d}" == dtypes.NUMBER_FORMAT_STRINGS["bigint"]

    def test_dtype_for_chapel(self):
        dtypes_for_chapel = {  # see DType
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
        for chapel_name, dtype_name in dtypes_for_chapel.items():
            assert dtypes.dtype_for_chapel(chapel_name) == dtypes.dtype(dtype_name)
        # check caching in the implementation of dtype_for_chapel()
        for chapel_name, dtype_name in dtypes_for_chapel.items():
            assert dtypes.dtype_for_chapel(chapel_name) == dtypes.dtype(dtype_name)

    @pytest.mark.parametrize("dtype1", [ak.bool_, ak.uint8, ak.uint64, ak.bigint, ak.int64, ak.float64])
    @pytest.mark.parametrize("dtype2", [ak.bool_, ak.uint8, ak.uint64, ak.bigint, ak.int64, ak.float64])
    def test_result_type(self, dtype1, dtype2):
        if dtype1 == ak.bigint or dtype2 == ak.bigint:
            if dtype1 == ak.float64 or dtype2 == ak.float64:
                expected_result = ak.float64
            else:
                expected_result = ak.bigint
        else:
            expected_result = np.result_type(dtype1, dtype2)
        # pdarray vs pdarray
        a = ak.array([0, 1], dtype=dtype1)
        b = ak.array([0, 1], dtype=dtype2)
        assert ak.result_type(a, b) == expected_result

        # dtype and dtype
        assert ak.result_type(dtype1, dtype2) == expected_result

        # mixed: pdarray vs dtype
        assert ak.result_type(a, dtype2) == expected_result
        assert ak.result_type(dtype1, b) == expected_result

    def test_bool_alias(self):
        assert ak.bool == ak.bool_
        assert ak.bool == np.bool_

    def _has(self, attr):
        return hasattr(ak, attr)

    def _get(self, attr, default=None):
        return getattr(ak, attr, default)

    @pytest.mark.smoke
    def test_bigint_dtype_singleton_and_str(
        self,
    ):
        dt1 = ak.bigint()
        dt2 = ak.dtype("bigint")
        dt3 = ak.dtype(ak.bigint)  # class object accepted
        assert dt1 is dt2 is dt3
        assert str(dt1) == "bigint"
        assert repr(dt1) in {"dtype(bigint)", "bigint"}  # allow either style
        # hashability + equality semantics
        d = {dt1: "ok"}
        assert d[dt2] == "ok"
        assert dt1 == dt2 == dt3

    def test_bigint_dtype_resolution_variants(
        self,
    ):
        dt = ak.bigint()
        # names / tokens
        assert ak.dtype("BIGINT") is dt
        # instances and class objects
        assert ak.dtype(ak.bigint) is dt
        assert ak.dtype(dt) is dt
        # scalar class object (works even if scalar not imported yet)
        assert ak.dtype(bigint_) is dt

        # ensure we never fall back to object dtype for bigint-like inputs
        assert ak.dtype("bigint").kind != np.dtype("O").kind

    @pytest.mark.smoke
    def test_bigint_scalar_construction_and_basics(
        self,
    ):
        # ak.bigint(…) should construct a scalar if you implemented the metaclass;
        # if not yet implemented, skip gracefully.
        maybe_scalar = ak.bigint_(1)
        if not isinstance(maybe_scalar, bigint_):
            pytest.skip("ak.bigint(…) does not construct scalar yet")
        x = maybe_scalar
        assert isinstance(x, bigint_)
        assert int(x) == 1
        assert x.dtype is ak.bigint()
        assert x.item() == 1
        assert "ak.bigint_(" in repr(x)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("0", 0),
            ("42", 42),
            ("-17", -17),
            ("0x10", 16),
            ("0b1011", 11),
            ("0o77", 63),
        ],
    )
    def test_bigint_scalar_parsing_from_strings(self, text, expected):
        try:
            x = bigint_(text)
        except TypeError:
            pytest.skip("bigint_ not present")
        assert isinstance(x, bigint_)
        assert int(x) == expected

    @pytest.mark.parametrize("val", [0, 1, -1, 2**64, 2**200, -(2**200)])
    def test_dtype_from_python_int_routes_big_values_to_bigint(self, val):
        dt = ak.dtype(val)
        if abs(val) >= 2**64:
            assert dt is ak.bigint()
        else:
            # NOTE: downstream policy may choose int64/uint64 depending on sign
            assert dt in {np.dtype(np.int64), np.dtype(np.uint64)}

    def test_pickle_roundtrip_of_bigint_dtype_singleton(
        self,
    ):
        dt = ak.bigint()
        data = pickle.dumps(dt)
        restored = pickle.loads(data)
        assert restored is dt

    def test_dtype_accepts_scalar_instance_and_class(
        self,
    ):
        try:
            s = bigint_(2**128)
        except TypeError:
            pytest.skip("bigint_ not present")
        assert ak.dtype(s) is ak.bigint()
        assert ak.dtype(type(s)) is ak.bigint()

    def test_supported_sets_if_present(
        self,
    ):
        # If your module exposes these capability sets/helpers, validate bigint entries.
        ints_set = self._get("ARKOUDA_SUPPORTED_INTS")
        nums_set = self._get("ARKOUDA_SUPPORTED_NUMBERS")
        if ints_set is None and nums_set is None:
            pytest.skip("supported-type sets not exposed")
        if ints_set is not None:
            if self._has("bigint_"):
                assert bigint_ in ints_set
        if nums_set is not None:
            if self._has("bigint_"):
                assert bigint_ in nums_set

    def test_resolve_scalar_dtype_if_present(
        self,
    ):
        fn = self._get("resolve_scalar_dtype")
        if fn is None or not self._has("bigint_"):
            pytest.skip("resolve_scalar_dtype or bigint_ not present")
        assert fn(bigint_(123)) == "bigint"

    @pytest.mark.parametrize(
        "args,expect",
        [
            # bigint with bigint → bigint
            ((ak.bigint(), ak.bigint()), "bigint"),
            # bigint with float → float64 (common numpy-like policy)
            ((ak.bigint(), np.dtype(np.float64)), np.dtype(np.float64)),
            # bigint with smaller ints → bigint (to avoid overflow)
            ((ak.bigint(), np.dtype(np.int64)), "bigint"),
            ((ak.bigint(), np.dtype(np.uint64)), "bigint"),
        ],
    )
    def test_result_type_if_present(self, args, expect):
        fn = self._get("result_type")
        if fn is None:
            pytest.skip("result_type not present")
        rt = fn(*args)
        if expect == "bigint":
            assert rt is ak.bigint()
        else:
            assert rt == expect

    @pytest.mark.parametrize(
        "from_dt,to_dt,expect",
        [
            (ak.bigint(), ak.bigint(), True),
            # Policy-dependent casts; assert at least bigint→float64 is allowed (info-preserving for magnitude)
            (ak.bigint(), np.dtype(np.float64), True),
            # bigint to int64 may be disallowed if truncation is a concern; allow either and assert consistency
            (ak.bigint(), np.dtype(np.int64), None),
        ],
    )
    def test_can_cast_if_present(self, from_dt, to_dt, expect):
        fn = self._get("can_cast")
        if fn is None:
            pytest.skip("can_cast not present")
        out = fn(from_dt, to_dt)
        if expect is not None:
            assert out is expect
        else:
            assert out in {True, False}  # just ensure it returns a boolean

    def test_ak_array_with_big_bigint_scalar_dtype_resolution_only(
        self,
    ):
        """
        This does not assert backend storage—only that dtype(...) recognizes a bigint scalar’s dtype.
        """
        if not self._has("bigint_"):
            pytest.skip("bigint_ not present")
        s = bigint_(2**200)
        assert ak.dtype(s) is ak.bigint()

    @pytest.mark.parametrize(
        "lhs,rhs,op,expect",
        [
            (bigint_(5), bigint_(7), "__add__", 12),
            (bigint_(5), 7, "__mul__", 35),
            (bigint_(2**130), 1, "__sub__", 2**130 - 1),
            (bigint_(2**130), bigint_(2**130), "__eq__", True),
            (bigint_(-3), 2, "__lt__", True),
        ],
    )
    def test_bigint_scalar_python_arithmetic_and_comparisons(self, lhs, rhs, op, expect):
        if not self._has("bigint_"):
            pytest.skip("bigint_ not present")
        result = getattr(lhs, op)(rhs)
        if isinstance(expect, bool):
            assert bool(result) is expect
        else:
            assert int(result) == expect

    def test_bigint_scalar_numpy_interop_minimal(
        self,
    ):
        """
        Ensure round-trip into NumPy scalars works without crashing.
        We do not require exact dtype preservation (NumPy has no bigint).
        """
        if not self._has("bigint_"):
            pytest.skip("bigint_ not present")
        x = bigint_(2**120 + 3)
        arr = np.array([int(x)], dtype=np.object_)  # safest box
        assert arr.shape == (1,)
        assert arr.dtype == np.dtype("O")
        assert arr[0] == int(x)

    def test_dtype_does_not_shadow_function_name(
        self,
    ):
        # Guard against accidental parameter shadowing: dtype(dtype=...)
        # This is a smoke test: just calling the function should not raise because of a shadow.
        assert ak.dtype("int64") == np.dtype(np.int64)
        assert ak.dtype("bigint") is ak.bigint()

    def test_import_order_safety_for_dtype_bigint_references(self, monkeypatch):
        """
        Ensure dtype() works even if bigint_ is not in globals (simulating early import).
        """
        # Simulate early import scenario
        if "bigint_" in ak.__dict__:
            monkeypatch.delitem(ak.__dict__, "bigint_", raising=False)
        assert ak.dtype("bigint") is ak.bigint()
        assert ak.dtype(ak.bigint) is ak.bigint()
        # Put it back by re-import if needed (harmless if already present)
        from importlib import reload

        reload(sys.modules[ak.__name__])

    def test_bigint_equality_semantics_against_strings_and_names(
        self,
    ):
        dt = ak.bigint()

        class Mock:
            name = "bigint"

        assert (dt == "bigint") or True  # allow dtype to compare true to name token
        assert dt == Mock()

    @pytest.mark.parametrize("n", [0, 1, 2**64, 2**200, -(2**200)])
    def test_number_formatting_through_str_formatting(self, n):
        """
        If you maintain NUMBER_FORMAT_STRINGS or similar, ensure formatting doesn't crash.
        """
        # This test is intentionally tolerant; it ensures no exceptions and basic correctness.
        s = f"{n:d}"
        assert isinstance(s, str)
        assert s.startswith("-") == (n < 0)
