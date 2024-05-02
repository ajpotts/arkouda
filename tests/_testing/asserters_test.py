import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import cast
from arkouda import float64 as akfloat64
from arkouda.numpy import nan
from arkouda.testing import (
    assert_arkouda_array_equal,
    assert_attr_equal,
    assert_class_equal,
    assert_index_equal,
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

        self.assertIsInstance(idx, ak.Index)
        self.assertEqual(idx.size, 5)
        self.assertListEqual(idx.to_list(), [i for i in range(5)])

        a = pd.Index([1, 2, 3])
        # assert_index_equal(a, a)
        assert_index_equal(idx, idx)

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

    # @ TODO Complete
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

    def test_assert_arkouda_array_equal(self):
        size = 5
        a = ak.arange(size)
        a1 = -1 * ak.arange(size)
        b = ak.array([1, 2, nan, 3, 4, nan])
        a_float = cast(a, dt=akfloat64)

        assert_arkouda_array_equal(a, a)
        assert_arkouda_array_equal(a, a, index_values=a)
        assert_arkouda_array_equal(b, b)
        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(a, a1)

        #   check_dtype
        assert_arkouda_array_equal(a, a_float, check_dtype=False)
        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(a, a_float, check_dtype=True)

        #   check_same
        a_copy = a[:]
        assert_arkouda_array_equal(a, a_copy)
        assert_arkouda_array_equal(a, a, check_same="same")
        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(a, a, check_same="copy")
        assert_arkouda_array_equal(a, a_copy, check_same="copy")
        with self.assertRaises(AssertionError):
            assert_arkouda_array_equal(a, a_copy, check_same="same")

    # @ TODO Complete
    def test_assert_dict_equal(self):
        pass

    ###################################

    # @ TODO Complete
    def test_assert_is_valid_plot_return_object(self):
        pass

    # @ TODO Complete
    def test_assert_is_sorted(self):
        pass

    # @ TODO Complete
    def test_assert_categorical_equal(self):
        pass

    # @ TODO Complete
    def test_assert_series_equal(self):
        pass

    # @ TODO Complete
    def test_assert_frame_equal(self):
        pass

    # @ TODO Complete
    def test_assert_equal(self):
        pass

    # @ TODO Complete
    def test_assert_contains_all(self):
        pass

    # @ TODO Complete
    def test_assert_copy(self):
        pass

    # @ TODO Complete
    def test_assert_indexing_slices_equivalent(self):
        pass

    # @ TODO Complete
    def test_assert_metadata_equivalent(self):
        pass
