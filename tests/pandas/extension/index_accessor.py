import pandas as pd

import arkouda as ak

from arkouda.index import Index as ak_Index
from arkouda.index import MultiIndex as ak_MultiIndex
from arkouda.pandas.extension import ArkoudaExtensionArray
from arkouda.pandas.extension._index_accessor import _ak_index_to_pandas_no_copy, _pandas_index_to_ak


def _assert_index_equal_values(p_idx: pd.Index, values):
    """Helper: assert pandas Index values == iterable `values`."""
    assert list(p_idx.tolist()) == list(values)


class TestArkoudaIndexAccessor:
    def test_index_accessor_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _index_accessor

        result = doctest.testmod(
            _index_accessor, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_to_ak_simple_index_roundtrip(self):
        idx = pd.Index([1, 2, 3], name="id")

        # Initially not Arkouda-backed
        assert not idx.ak.is_arkouda

        # Convert to Arkouda-backed pandas.Index
        idx_ak = idx.ak.to_ak()
        assert isinstance(idx_ak, pd.Index)
        assert idx_ak.name == "id"
        assert idx_ak.ak.is_arkouda

        # Underlying array is ArkoudaExtensionArray
        arr = idx_ak.array
        assert isinstance(arr, ArkoudaExtensionArray)
        akcol = arr._data
        assert isinstance(akcol, ak.pdarray)
        _assert_index_equal_values(idx_ak, akcol.tolist())

        # Collect back to plain NumPy-backed Index
        idx_local = idx_ak.ak.collect()
        assert isinstance(idx_local, pd.Index)
        assert not isinstance(idx_local.array, ArkoudaExtensionArray)
        assert not idx_local.ak.is_arkouda
        assert idx_local.name == "id"
        assert idx_local.equals(idx)

    def test_collect_on_plain_index_is_noop_semantics(self):
        """collect() on a non-Arkouda Index should behave like a simple copy."""
        idx = pd.Index([10, 20, 30], name="nums")
        assert not idx.ak.is_arkouda

        idx2 = idx.ak.collect()
        assert isinstance(idx2, pd.Index)
        assert not isinstance(idx2.array, ArkoudaExtensionArray)
        assert not idx2.ak.is_arkouda
        assert idx2.equals(idx)
        assert idx2.name == "nums"

    def test_to_ak_legacy_simple_index_and_helper_roundtrip(self):
        idx = pd.Index([4, 5, 6], name="foo")

        # pandas -> ak.Index
        ak_idx = idx.ak.to_ak_legacy()
        assert isinstance(ak_idx, ak_Index)
        assert ak_idx.name == "foo"
        assert ak_idx.size == len(idx)
        assert ak_idx.values.tolist() == [4, 5, 6]

        # ak.Index -> pandas via helper (zero-copy-ish EA)
        idx_ea = _ak_index_to_pandas_no_copy(ak_idx)
        assert isinstance(idx_ea, pd.Index)
        assert isinstance(idx_ea.array, ArkoudaExtensionArray)
        assert idx_ea.name == "foo"
        _assert_index_equal_values(idx_ea, [4, 5, 6])

    def test_pandas_index_to_ak_and_back(self):
        idx = pd.Index(["a", "b", "c"], name="letters")

        # pandas.Index -> ak.Index
        ak_idx = _pandas_index_to_ak(idx)
        assert isinstance(ak_idx, ak_Index)
        assert ak_idx.name == "letters"
        assert ak_idx.size == 3
        assert ak_idx.values.tolist() == ["a", "b", "c"]

        # ak.Index -> pandas.Index (EA-backed)
        idx_back = _ak_index_to_pandas_no_copy(ak_idx)
        assert isinstance(idx_back, pd.Index)
        assert isinstance(idx_back.array, ArkoudaExtensionArray)
        assert idx_back.name == "letters"
        _assert_index_equal_values(idx_back, ["a", "b", "c"])

    def test_to_ak_multiindex_roundtrip(self):
        arrays = [[1, 2, 3], ["a", "b", "c"]]
        midx = pd.MultiIndex.from_arrays(arrays, names=["num", "char"])

        assert not midx.ak.is_arkouda

        # pandas.MultiIndex -> Arkouda-backed MultiIndex (pandas object)
        midx_ak = midx.ak.to_ak()
        assert isinstance(midx_ak, pd.MultiIndex)
        assert list(midx_ak.names) == ["num", "char"]
        assert midx_ak.ak.is_arkouda

        # Each level should be ArkoudaExtensionArray-backed
        for lvl in midx_ak.levels:
            assert isinstance(lvl.array, ArkoudaExtensionArray)

        # Collect back to plain NumPy-backed MultiIndex
        midx_local = midx_ak.ak.collect()
        assert isinstance(midx_local, pd.MultiIndex)
        assert not midx_local.ak.is_arkouda
        assert midx_local.equals(midx)

        # Levels are no longer ArkoudaExtensionArray-backed
        for lvl in midx_local.levels:
            assert not isinstance(lvl.array, ArkoudaExtensionArray)

    def test_to_ak_legacy_multiindex_and_helper_roundtrip(self):
        arrays = [[1, 1, 2], ["red", "blue", "red"]]
        midx = pd.MultiIndex.from_arrays(arrays, names=["num", "color"])

        # pandas.MultiIndex -> ak.MultiIndex
        ak_midx = midx.ak.to_ak_legacy()
        assert isinstance(ak_midx, ak_MultiIndex)
        assert ak_midx.nlevels == 2
        assert list(ak_midx.names) == ["num", "color"]
        assert ak_midx.size == len(midx)

        # ak.MultiIndex -> pandas.MultiIndex with EA-backed levels
        midx_ea = _ak_index_to_pandas_no_copy(ak_midx)
        assert isinstance(midx_ea, pd.MultiIndex)
        assert list(midx_ea.names) == ["num", "color"]
        for lvl in midx_ea.levels:
            assert isinstance(lvl.array, ArkoudaExtensionArray)

        # Values should match original (up to pandas' internal coding)
        assert list(midx_ea) == list(midx)

    def test_is_arkouda_flag_index_and_multiindex(self):
        idx = pd.Index([1, 2, 3])
        midx = pd.MultiIndex.from_arrays([[1, 2], ["a", "b"]])

        # Initially false
        assert not idx.ak.is_arkouda
        assert not midx.ak.is_arkouda

        idx_ak = idx.ak.to_ak()
        midx_ak = midx.ak.to_ak()

        assert idx_ak.ak.is_arkouda
        assert midx_ak.ak.is_arkouda

        idx_local = idx_ak.ak.collect()
        midx_local = midx_ak.ak.collect()

        assert not idx_local.ak.is_arkouda
        assert not midx_local.ak.is_arkouda

    def test_from_ak_legacy_roundtrip_index(self):
        """Round-trip using to_ak_legacy() + from_ak_legacy()."""
        idx = pd.Index([7, 8, 9], name="nums")

        ak_idx = idx.ak.to_ak_legacy()
        assert isinstance(ak_idx, ak_Index)

        # NOTE: from_ak_legacy should *not* call akidx.to_ak();
        # it should just wrap via _ak_index_to_pandas_no_copy.
        idx_back = ak.pandas.extension.ArkoudaIndexAccessor.from_ak_legacy(ak_idx)
        assert isinstance(idx_back, pd.Index)
        assert isinstance(idx_back.array, ArkoudaExtensionArray)
        assert idx_back.name == "nums"
        _assert_index_equal_values(idx_back, [7, 8, 9])

    def test_concat_simple_index_via_accessor(self):
        # plain pandas indices
        left = pd.Index([1, 2, 3], name="id")
        right = pd.Index([4, 5], name="id")

        # Run through the .ak accessor (no need to call to_ak() explicitly;
        # concat should lift to legacy and back internally).
        out = left.ak.concat(right)

        # Result is a pandas Index, Arkouda-backed
        assert isinstance(out, pd.Index)
        assert out.ak.is_arkouda
        assert out.name == "id"

        # Values are concatenated in order
        _assert_index_equal_values(out, [1, 2, 3, 4, 5])

        # Collect should give us a plain NumPy-backed Index with same values
        collected = out.ak.collect()
        assert isinstance(collected, pd.Index)
        assert not collected.ak.is_arkouda
        assert collected.equals(pd.Index([1, 2, 3, 4, 5], name="id"))

    def test_concat_multiindex_via_accessor(self):
        arrays_left = [[1, 1, 2], ["red", "blue", "red"]]
        arrays_right = [[3, 3], ["green", "red"]]

        left = pd.MultiIndex.from_arrays(arrays_left, names=["num", "color"])
        right = pd.MultiIndex.from_arrays(arrays_right, names=["num", "color"])

        out = left.ak.concat(right)

        # Result is a pandas.MultiIndex, Arkouda-backed
        assert isinstance(out, pd.MultiIndex)
        assert out.ak.is_arkouda
        assert out.names == ["num", "color"]

        # Order-preserving concat of the tuples
        expected_tuples = list(left.tolist()) + list(right.tolist())
        assert list(out.tolist()) == expected_tuples

        # Collect should give a plain MultiIndex with same tuples / names
        collected = out.ak.collect()
        assert isinstance(collected, pd.MultiIndex)
        assert not collected.ak.is_arkouda
        assert list(collected.tolist()) == expected_tuples
        assert collected.names == ["num", "color"]

    def test_lookup_simple_index_scalar_and_array_via_accessor(self):
        idx = pd.Index([10, 20, 30], name="nums")
        ak_idx = idx.ak.to_ak()

        # Scalar lookup should be handled by wrapping it into a one-element pdarray
        mask_scalar = ak_idx.ak.lookup(20)
        import arkouda as ak  # local import to avoid confusion with other modules

        assert isinstance(mask_scalar, ak.pdarray)
        assert mask_scalar.tolist() == [False, True, False]

        # Array lookup should be passed through directly
        keys = ak.array([5, 10, 30])
        mask_array = ak_idx.ak.lookup(keys)
        assert isinstance(mask_array, ak.pdarray)
        # in1d(self.values, [5, 10, 30]) → [10, 30] are present
        assert mask_array.tolist() == [True, False, True]

    def test_lookup_multiindex_tuple_key_via_accessor(self):
        arrays = [[1, 1, 2, 3], ["red", "blue", "red", "red"]]
        midx = pd.MultiIndex.from_arrays(arrays, names=["num", "color"])
        ak_midx = midx.ak.to_ak()

        # Tuple key corresponds to one row in the MultiIndex
        mask = ak_midx.ak.lookup((1, "red"))

        import arkouda as ak

        assert isinstance(mask, ak.pdarray)
        # Only the first row (1, "red") should match
        assert mask.to_list() == [True, False, False, False]

        # A key that matches multiple rows
        mask_multi = ak_midx.ak.lookup((2, "red"))
        assert isinstance(mask_multi, ak.pdarray)
        # Only the third row (2, "red") matches here
        assert mask_multi.to_list() == [False, False, True, False]
