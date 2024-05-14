from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import Categorical, DataFrame, Index, MultiIndex, Series, Strings, cast
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
    assert_is_sorted,
    assert_series_equal,
)


class AssertersTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(AssertersTest, cls).setUpClass()

    def build_index(self) -> Index:
        idx = ak.Index(ak.arange(5), name="test1")
        return idx

    def build_multi_index(self) -> MultiIndex:
        midx = ak.MultiIndex([ak.arange(5), -1 * ak.arange(5)], names=["test1", "test2"])
        return midx

    def build_ak_df(self, index_dtype="int64", index_name=None) -> DataFrame:
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
            },
            index=Index(ak.arange(6, dtype=index_dtype), name=index_name),
        )

    def test_assert_almost_equal(self):
        size = 10

        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001
        a = ak.arange(size, dtype="float64")
        a2 = a + rtol * a + atol * rng.random()
        a3 = a + rtol + atol

        assert_almost_equal(a, a2, atol=atol, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_almost_equal(a, a3, atol=atol, rtol=rtol)

        idx = Index(a)
        idx2 = Index(a2)
        idx3 = Index(a3)

        assert_almost_equal(idx, idx2, atol=atol, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_almost_equal(idx, idx3, atol=atol, rtol=rtol)

        s = Series(a)
        s2 = Series(a2)
        s3 = Series(a3)

        assert_almost_equal(s, s2, atol=atol, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_almost_equal(s, s3, atol=atol, rtol=rtol)

        df = DataFrame({"col1": a}, index=idx)
        df2 = DataFrame({"col1": a2}, index=idx2)
        df3 = DataFrame({"col1": a3}, index=idx3)

        assert_almost_equal(df, df2, atol=atol, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_almost_equal(df, df3, atol=atol, rtol=rtol)

        assert_almost_equal(True, True, atol=atol, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_almost_equal(True, False, atol=atol, rtol=rtol)

        assert_almost_equal(1.0, 1.0, atol=atol, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_almost_equal(1.0, 1.5, atol=atol, rtol=rtol)

    def test_assert_index_equal(self):
        size = 10

        # exact
        i1 = Index(ak.arange(size, dtype="float64"))
        i2 = Index(ak.arange(size, dtype="int64"))
        assert_index_equal(i1, i2, exact=False)
        with self.assertRaises(AssertionError):
            assert_index_equal(i1, i2, exact=True)

        # check_names
        i3 = Index(ak.arange(size), name="name1")
        i4 = Index(ak.arange(size), name="name1")
        i5 = Index(ak.arange(size), name="name2")

        assert_index_equal(i3, i4, check_names=True)
        assert_index_equal(i3, i5, check_names=False)
        with self.assertRaises(AssertionError):
            assert_index_equal(i3, i5, check_names=True)

    def test_assert_index_equal_categorical(self):
        # check_categorical
        # check_order
        i1 = Index(Categorical(ak.array(["a", "a", "b"])))
        i3 = Index(Categorical(ak.array(["a", "b", "a"])))
        i4 = Index(Categorical(ak.array(["a", "b", "c"])))
        i5 = Index(Categorical(ak.array(["a", "a", "b"])).sort())

        assert_index_equal(i1, i1)
        assert_index_equal(i1, i3, check_order=False)
        with self.assertRaises(AssertionError):
            assert_index_equal(i1, i3, check_order=True)
        with self.assertRaises(AssertionError):
            assert_index_equal(i1, i3, check_categorical=False)
        with self.assertRaises(AssertionError):
            assert_index_equal(i1, i4, check_categorical=False)
        assert_index_equal(i1, i5, check_order=True, check_categorical=True)

    def test_assert_index_equal_check_exact(self):
        size = 10

        # check_exact
        i1 = Index(ak.arange(size, dtype="float64"))
        i2 = Index(ak.arange(size) + 1e-9)
        assert_index_equal(i1, i2, check_exact=False)
        with self.assertRaises(AssertionError):
            assert_index_equal(i1, i2, check_exact=True)

        # rtol
        # atol
        i3_float = Index(ak.arange(size, dtype="float64"))

        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001

        i3_atol = Index(ak.arange(size) + atol * rng.random())
        assert_index_equal(i3_float, i3_atol, check_exact=False, atol=atol)

        i3_atol_rtol = Index(ak.arange(size) + rtol * ak.arange(size) + atol * rng.random())
        assert_index_equal(i3_float, i3_atol_rtol, check_exact=False, atol=atol, rtol=rtol)

        i3_2rtol = Index(ak.arange(size) + ak.arange(size) * 2 * rtol)
        with self.assertRaises(AssertionError):
            assert_index_equal(i3_float, i3_2rtol, check_exact=False, rtol=rtol)

        i3_2atol = Index(ak.arange(size) + 2 * atol)
        with self.assertRaises(AssertionError):
            assert_index_equal(i3_float, i3_2atol, check_exact=False, atol=atol)

    def test_assert_index_equal_multiindex(self):
        m1 = self.build_multi_index()
        m2 = self.build_multi_index()

        assert_index_equal(m1, m2)

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
        s = ak.Series(-1 * ak.arange(5), index=ak.arange(5))

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

        from arkouda.testing import _check_isinstance

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
        size = 10
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
        size = 10
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

    def test_assert_categorical_equal(self):
        c3 = Categorical(ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"]))
        c4 = Categorical(ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])).sort()
        assert_categorical_equal(c3, c4, check_category_order=False)
        with self.assertRaises(AssertionError):
            assert_categorical_equal(c3, c4, check_category_order=True)

    def test_assert_series_equal_check_names(self):
        s = Series(ak.array(["a", "b", "c"]), index=Index(ak.arange(3)), name="test")
        assert_series_equal(s, s)

        # check_names
        s_diff_name = Series(ak.array(["a", "b", "c"]), index=Index(ak.arange(3)), name="different_name")
        assert_series_equal(s, s_diff_name, check_names=False)
        with self.assertRaises(AssertionError):
            assert_series_equal(s, s_diff_name, check_names=True)

    def test_assert_series_equal(self):
        s = Series(ak.array([1, 0, 2]), index=Index(ak.arange(3)))
        s_float = Series(ak.array([1.0, 0.0, 2.0]), index=Index(ak.arange(3) * 1.0))

        assert_series_equal(s, s)
        assert_series_equal(s_float, s_float)

        #   check_dtype
        assert_series_equal(s, s_float, check_dtype=False, check_index_type=False)
        with self.assertRaises(AssertionError):
            assert_series_equal(s, s_float, check_dtype=False, check_index_type=True)
        with self.assertRaises(AssertionError):
            assert_series_equal(s, s_float, check_dtype=True, check_index_type=False)

        # check_index
        s_diff_index = Series(ak.array([1, 0, 2]), index=Index(ak.arange(3) * 2.0))
        assert_series_equal(s, s_diff_index, check_index=False)
        with self.assertRaises(AssertionError):
            assert_series_equal(s, s_diff_index, check_index=True)

        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001
        s_atol = Series(
            ak.array([1, 0, 2]) + rng.random() * atol, index=Index(ak.arange(3) + rng.random() * atol)
        )

        diff_rtol_atol = rtol * ak.array([1, 0, 2]) + rng.random() * atol
        d2 = rtol * ak.arange(3) + rng.random() * atol

        s_rtol_atol = Series(
            ak.array([1, 0, 2]) + diff_rtol_atol,
            index=Index(ak.arange(3) + d2),
        )

        s_2rtol = Series(
            ak.array([1, 0, 2]) + ak.array([1, 0, 2]) * 2 * rtol,
            index=Index(ak.arange(3) + ak.array([1, 0, 2]) * 2 * rtol),
        )

        s_2atol = Series(
            ak.array([1, 0, 2]) + 2 * atol,
            index=Index(ak.arange(3) + 2 * atol),
        )

        assert_series_equal(s_float, s_atol, check_exact=False, atol=atol)
        assert_series_equal(s_float, s_rtol_atol, check_exact=False, atol=atol, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_series_equal(s_float, s_2rtol, check_exact=False, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_series_equal(s_float, s_2atol, check_exact=False, atol=atol)

    def test_assert_series_equal_check_like(self):
        # check_like
        s_unordered_index = Series(ak.array([1, 0, 2]), index=Index(ak.array([0, 2, 1])))
        s_ordered_index = s_unordered_index.sort_index()
        assert_series_equal(s_ordered_index, s_unordered_index, check_like=True)
        with self.assertRaises(AssertionError):
            assert_series_equal(s_ordered_index, s_unordered_index, check_like=False)

    def test_assert_series_equal_categorical(self):
        # check_categorical
        # check_category_order

        s3a = Series(
            Categorical(ak.array(["a", "b", "c"])),
            index=Index(Categorical(ak.array(["a", "a", "b"]))),
            name="test",
        )
        s3b = Series(
            Categorical(ak.array(["a", "b", "c"])).sort(),
            index=Index(Categorical(ak.array(["a", "a", "b"]))),
            name="test",
        )
        assert_series_equal(s3a, s3a)
        with self.assertRaises(AssertionError):
            assert_series_equal(s3a, s3b, check_categorical=True, check_category_order=True)
        assert_series_equal(s3a, s3b, check_categorical=True, check_category_order=False)

    def test_assert_frame_equal(self):
        df = self.build_ak_df()
        df2 = self.build_ak_df()
        assert_frame_equal(df, df2)

    def test_assert_frame_equal_check_dtype(self):
        df = self.build_ak_df()

        # check_dtype
        df_cpy = df.copy(deep=True)
        assert_frame_equal(df, df_cpy, check_dtype=True)
        df_cpy["day"] = cast(df_cpy["day"], dt="float64")
        assert_frame_equal(df_cpy, df_cpy, check_dtype=True)
        assert_frame_equal(df, df_cpy, check_dtype=False)
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_cpy, check_dtype=True)

    def test_assert_frame_equal_check_index_type(self):
        df = self.build_ak_df()

        # check_index_type
        df_float_index = self.build_ak_df(index_dtype="float64")
        assert_frame_equal(df, df_float_index, check_index_type=False)
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_float_index, check_index_type=True)

    def test_assert_frame_equal_check_names(self):
        # check_names
        df_name1 = self.build_ak_df(index_name="name1")
        df_name2 = self.build_ak_df(index_name="name2")
        assert_frame_equal(df_name1, df_name2, check_names=False)
        with self.assertRaises(AssertionError):
            assert_frame_equal(df_name1, df_name2, check_names=True)

    def test_assert_frame_equal_check_like(self):
        df = self.build_ak_df()

        # check_like
        df_sorted = df.sort_values("amount")
        assert_frame_equal(df, df_sorted, check_like=True)
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_sorted, check_like=False)

        df_new_col_order = df[["bi", "userID", "day", "item", "amount", "userName"]]
        assert_frame_equal(df, df_new_col_order, check_like=True)
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_new_col_order, check_column_type=True)

    def test_assert_frame_equal_check_categorical(self):
        # check_categorical
        df = self.build_ak_df()
        df["userName"] = Categorical(df["userName"])
        df_ordered = self.build_ak_df()
        df_ordered["userName"] = Categorical(df_ordered["userName"]).sort()

        assert_frame_equal(df, df_ordered, check_categorical=False)
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_ordered, check_categorical=True)

    def test_assert_frame_equal_check_exact(self):
        # check_exact
        # rtol
        # atol
        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001

        df = self.build_ak_df()
        df_rtol_atol = self.build_ak_df()
        df_rtol_atol["amount"] = (
            df_rtol_atol["amount"] + rtol * df_rtol_atol["amount"] + rng.random() * atol
        )

        assert_frame_equal(df, df_rtol_atol, check_exact=False, atol=atol, rtol=rtol)

        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_rtol_atol, check_exact=True)
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_rtol_atol, check_exact=False, rtol=rtol)
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_rtol_atol, check_exact=False, atol=atol)

    def test_assert_equal(self):
        size = 10
        a = ak.arange(size)
        a2 = a + 1
        idx = Index(a)
        idx2 = Index(a2)
        s = Series(a)
        s2 = Series(a2)
        df = DataFrame({"col": a}, index=idx)
        df2 = DataFrame({"col": a2}, index=idx2)

        assert_equal(a, a)
        with self.assertRaises(AssertionError):
            assert_equal(a, a2)

        assert_equal(idx, idx)
        with self.assertRaises(AssertionError):
            assert_equal(idx, idx2)

        assert_equal(s, s)
        with self.assertRaises(AssertionError):
            assert_equal(s, s2)

        assert_equal(df, df)
        with self.assertRaises(AssertionError):
            assert_equal(df, df2)

        st = "string1"
        st2 = "string2"
        assert_equal(st, st)
        with self.assertRaises(AssertionError):
            assert_equal(st, st2)

        n = 1.0
        n2 = 1.5
        assert_equal(n, n)
        with self.assertRaises(AssertionError):
            assert_equal(n, n2)

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

    def test_assert_arkouda_array_equal(self):
        size = 10
        a = ak.arange(size)
        a2 = a + 1
        assert_arkouda_array_equal(a, a)
        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(a, a2)

        s = ak.array(["a", "b", "b"])
        s2 = ak.array(["a", "b", "c"])
        assert_arkouda_array_equal(s, s)
        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(s, s2)

        c = Categorical(s)
        c2 = Categorical(s2)
        assert_arkouda_array_equal(c, c)
        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(c, c2)

        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(a, s)

        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(s, c)

    def test_assert_arkouda_array_equal_bigint(self):
        size = 10
        a = ak.arange(size, dtype=ak.bigint) + (2**64 - size - 1)
        a2 = a + 1
        assert_arkouda_array_equal(a, a)
        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(a, a2)
