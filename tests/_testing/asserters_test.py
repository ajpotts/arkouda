import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.testing import assert_index_equal, assert_attr_equal


class AssertersTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(AssertersTest, cls).setUpClass()

    def test_assert_almost_equal(self):
        idx = ak.Index(ak.arange(5))

        self.assertIsInstance(idx, ak.Index)
        self.assertEqual(idx.size, 5)
        self.assertListEqual(idx.to_list(), [i for i in range(5)])

        a = pd.Index([1, 2, 3])
        # assert_index_equal(a, a)
        assert_index_equal(idx, idx)

    def test_assert_attr_equal(self):
        idx = ak.Index(ak.arange(5))

        assert_attr_equal("name", idx, idx, obj="Index")

    def test_assert_class_equal(self):
        pass

    def test_check_isinstance(self):
        pass

    def test_check_types(self):
        pass

    def test_assert_numpy_array_equal(self):
        pass

    def test_raise_assert_detail(self):
        pass

    ###################################

    def test_assert_is_valid_plot_return_object(self):
        pass

    def test_assert_is_sorted(self):
        pass

    def test_assert_categorical_equal(self):
        pass

    def test_assert_series_equal(self):
        pass

    def test_assert_frame_equal(self):
        pass

    def test_assert_equal(self):
        pass

    def test_assert_contains_all(self):
        pass

    def test_assert_copy(self):
        pass

    def test_assert_indexing_slices_equivalent(self):
        pass

    def test_assert_metadata_equivalent(self):
        pass
