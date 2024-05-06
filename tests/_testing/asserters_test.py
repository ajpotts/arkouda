import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import Categorical, DataFrame, Index, MultiIndex, Series, Strings, cast
from arkouda import float64 as akfloat64
from arkouda.numpy import nan
from arkouda.testing import (
    assert_almost_equal,
    assert_arkouda_array_equal,
    assert_arkouda_strings_equal,
    assert_attr_equal,
    assert_categorical_equal,
    assert_class_equal,
    assert_contains_all,
    assert_copy,
    assert_dict_equal,
    assert_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_indexing_slices_equivalent,
    assert_is_sorted,
    assert_series_equal,
)


class AssertersTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(AssertersTest, cls).setUpClass()

    def build_index(self):
        idx = ak.Index(ak.arange(5), name="test1")
        return idx

    def build_multi_index(self):
        midx = ak.MultiIndex([ak.arange(5), -1 * ak.arange(5)], names=["test1", "test2"])
        return midx

    def build_series(self):
        s = ak.Series(-1 * ak.arange(5), index=ak.arange(5))
        return s

    def build_ak_df(self):
        username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        bi = ak.arange(2**200, 2**200 + 6)
        return ak.DataFrame(
            {
                "userName": username,
                "userID": userid,
                "item": item,
                "day": day,
                "amount": amount,
                "bi": bi,
            }
        )

    # @ TODO Complete
    def test_assert_almost_equal(self):
        idx = ak.Index(ak.arange(5))

        a = pd.Index([1, 2, 3])
        # assert_index_equal(a, a)
        assert_index_equal(idx, idx)

    def test_assert_index_equal(self):
        i1 = Index(Categorical(ak.array(["a", "a", "b"])))
        i2 =Index(Categorical(ak.array(["a", "b", "a"])))

        assert_index_equal(i1, i1)
        assert_index_equal(i1, i2, check_categorical=False)

    def test_assert_attr_equal_index(self):
        idx = self.build_index()
        idx2 = self.build_index()

        assert_attr_equal("name", idx, idx2, obj="Index")
        assert_attr_equal("names", idx, idx2, obj="Index")
        assert_attr_equal("max_list_size", idx, idx2, obj="Index")

        idx2.name = "test2"
        with self.assertRaises(AssertionError):
            assert_attr_equal("name", idx, idx2, obj="Index")
        with self.assertRaises(AssertionError):
            assert_attr_equal("names", idx, idx2, obj="Index")

    def test_assert_attr_equal_multiindex(self):
        idx = self.build_index()
        midx = self.build_multi_index()
        midx2 = self.build_multi_index()

        assert_attr_equal("names", midx, midx2, obj="MultiIndex")

        midx3 = ak.MultiIndex([ak.arange(5), -1 * ak.arange(5)], names=["test1", "test3"])
        with self.assertRaises(AssertionError):
            assert_attr_equal("names", midx, midx3, obj="Index")
        with self.assertRaises(AssertionError):
            assert_attr_equal("names", idx, midx, obj="Index")

        assert_attr_equal("nlevels", midx, midx2, obj="MultiIndex")

    def test_assert_class_equal(self):
        idx = self.build_index()
        midx = self.build_multi_index()
        midx2 = self.build_multi_index()
        df = self.build_ak_df()
        s = self.build_series()

        assert_class_equal(idx, idx)
        assert_class_equal(midx, midx2)
        assert_class_equal(s, s)
        assert_class_equal(df, df)
        with self.assertRaises(AssertionError):
            assert_class_equal(midx, idx)
        with self.assertRaises(AssertionError):
            assert_class_equal(s, idx)
        with self.assertRaises(AssertionError):
            assert_class_equal(df, s)

    def test_assert_arkouda_strings_equal(self):
        a = ak.array(["a", "a", "b", "c"])
        a2 = ak.array(["a", "d", "b", "c"])
        a3 = ak.array(["a", "a", "b", "c", "d"])

        from arkouda.testing import _check_isinstance, assert_arkouda_strings_equal

        _check_isinstance(a, a, Strings)

        assert_arkouda_strings_equal(a, a)
        assert_arkouda_strings_equal(a, a, index_values=ak.arange(4))
        with self.assertRaises(AssertionError):
            assert_arkouda_strings_equal(a, a2)
        with self.assertRaises(AssertionError):
            assert_arkouda_strings_equal(a, a3)

        #   check_same
        a_copy = a[:]
        assert_arkouda_strings_equal(a, a_copy)
        assert_arkouda_strings_equal(a, a, check_same="same")
        with self.assertRaises(AssertionError):
            assert_arkouda_strings_equal(a, a, check_same="copy")
        assert_arkouda_strings_equal(a, a_copy, check_same="copy")
        with self.assertRaises(AssertionError):
            assert_arkouda_strings_equal(a, a_copy, check_same="same")

    def test_assert_dict_equal(self):
        size = 5
        dict1 = {"a": ak.arange(size), "b": -1 * ak.arange(size)}
        dict2 = {"a": ak.arange(size), "b": -1 * ak.arange(size)}
        dict3 = {"a": ak.arange(size), "c": -2 * ak.arange(size)}
        dict4 = {"a": ak.arange(size), "b": -1 * ak.arange(size), "c": -2 * ak.arange(size)}
        dict5 = {"a": ak.arange(size), "b": -2 * ak.arange(size)}

        assert_dict_equal(dict1, dict2)

        for d in [dict3, dict4, dict5]:
            with self.assertRaises(AssertionError):
                assert_dict_equal(dict1, d)

    def test_assert_is_sorted(self):
        size = 5
        a = ak.arange(size)
        b = -1 * a
        c = ak.array([1, 2, 5, 4, 3])

        assert_is_sorted(a)
        with self.assertRaises(AssertionError):
            assert_is_sorted(b)
        with self.assertRaises(AssertionError):
            assert_is_sorted(c)

        idx_a = Index(a)
        idx_b = Index(b)
        idx_c = Index(c)

        assert_is_sorted(idx_a)
        with self.assertRaises(AssertionError):
            assert_is_sorted(idx_b)
        with self.assertRaises(AssertionError):
            assert_is_sorted(idx_c)

        series_a = Series(a)
        series_b = Series(b)
        series_c = Series(c)

        assert_is_sorted(series_a)
        with self.assertRaises(AssertionError):
            assert_is_sorted(series_b)
        with self.assertRaises(AssertionError):
            assert_is_sorted(series_c)

    # @ TODO Complete
    def test_assert_categorical_equal(self):
        c1 = Categorical(ak.array(["a", "a", "b"]))
        c2 = Categorical(ak.array(["a", "b", "a"]))

        assert_categorical_equal(c1, c1)
        assert_categorical_equal(c1, c2, check_category_order=False)
        with self.assertRaises(AssertionError):
            assert_categorical_equal(c1, c2, check_category_order=True)

    # @ TODO Complete
    def test_assert_series_equal(self):
        s = Series(ak.array(["a", "b", "c"]), index=Index(ak.arange(3)), name="test")
        s2 = Series(ak.array([1, 0, 2]), index=Index(ak.arange(3)))
        s2_float = Series(ak.array([1.0, 0.0, 2.0]), index=Index(ak.arange(3) * 1.0))
        s_diff_name = Series(ak.array(["a", "b", "c"]), index=Index(ak.arange(3)), name="different_name")

        assert_series_equal(s, s)
        assert_series_equal(s2, s2)
        assert_series_equal(s2_float, s2_float)

        #   check_dtype
        assert_series_equal(s2, s2_float, check_dtype=False, check_index_type=False)
        with self.assertRaises(AssertionError):
            assert_series_equal(s2, s2_float, check_dtype=False, check_index_type=True)
        with self.assertRaises(AssertionError):
            assert_series_equal(s2, s2_float, check_dtype=True)

        # check_names
        assert_series_equal(s, s_diff_name, check_names=False)
        with self.assertRaises(AssertionError):
            assert_series_equal(s, s_diff_name, check_names=True)



        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001
        s2_random = Series(
            ak.array([1, 0, 2]) + rng.random() * atol, index=Index(ak.arange(3) + rng.random() * atol)
        )

        d1 = rtol * ak.array([1, 0, 2]) + rng.random() * atol
        d2 = rtol * ak.arange(3) + rng.random() * atol

        s2_random2 = Series(
            ak.array([1, 0, 2]) + d1,
            index=Index(ak.arange(3) + d2),
        )

        s2_random3 = Series(
            ak.array([1, 0, 2]) + ak.array([1, 0, 2]) * 2 * rtol,
            index=Index(ak.arange(3) + ak.array([1, 0, 2]) * 2 * rtol),
        )

        s2_random4 = Series(
            ak.array([1, 0, 2]) + 2 * atol,
            index=Index(ak.arange(3) + 2 * atol),
        )

        assert_series_equal(s2_float, s2_random, check_exact=False, atol=atol)
        assert_series_equal(s2_float, s2_random2, check_exact=False, atol=atol, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_series_equal(s2_float, s2_random3, check_exact=False, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_series_equal(s2_float, s2_random4, check_exact=False, atol=atol)

        # check_categorical: bool = True,
        # check_category_order: bool = True,

        s3a = Series(ak.array(["a", "b", "c"]), index=Index(Categorical(ak.array(["a", "a", "b"]))), name="test")
        s3b = Series(ak.array(["a", "b", "c"]), index=Index(Categorical(ak.array(["a", "b", "a"]))), name="test")
        assert_series_equal(s3a, s3a)
        with self.assertRaises(AssertionError):
            assert_series_equal(s3a, s3b)
        assert_series_equal(s3a, s3b, check_categorical=True, check_category_order=False)

        # check_index
        s2_diff_index = Series(ak.array([1, 0, 2]), index=Index(ak.arange(3) * 2.0))
        assert_series_equal(s2, s2_diff_index, check_index=False)
        with self.assertRaises(AssertionError):
            assert_series_equal(s2, s2_diff_index, check_index=True)

        # check_like
        s2_unordered_index = Series(ak.array([1, 0, 2]), index=Index(ak.array([0, 2, 1])))
        # assert_series_equal(s2, s2_unordered_index, check_like=True)
        with self.assertRaises(AssertionError):
            assert_series_equal(s2, s2_unordered_index, check_like=False)

    # @ TODO Complete
    def test_assert_frame_equal(self):
        pass

    # @ TODO Complete
    def test_assert_equal(self):
        pass

    # @ TODO Complete
    def test_assert_contains_all(self):
        d = {"a": 1, "b": 2, "c": 3}

        assert_contains_all([], d)
        assert_contains_all(["a", "b"], d)
        with self.assertRaises(AssertionError):
            assert_contains_all(["a", "d"], d)

    def test_assert_copy(self):
        size = 10
        df = DataFrame({"a": ak.arange(size), "b": -1 * ak.arange(size)})
        df_deep = df.copy(deep=True)
        df_shallow = df.copy(deep=False)

        cols = [df[col] for col in df.columns.values]
        cols_deep = [df_deep[col] for col in df_deep.columns.values]
        cols_shallow = [df_shallow[col] for col in df_shallow.columns.values]

        assert_copy(cols, cols_deep)

        with self.assertRaises(AssertionError):
            assert_copy(cols, cols_shallow)

    # @ TODO Complete
    def test_assert_indexing_slices_equivalent(self):
        pass