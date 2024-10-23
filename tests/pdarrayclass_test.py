import numpy as np
import pytest

import arkouda as ak
from arkouda.testing import assert_equal as ak_assert_equal

SEED = 314159
import arkouda.pdarrayclass
import numpy

REDUCTION_OPS = list(set(ak.pdarrayclass.supported_reduction_ops) - set(["isSorted", "isSortedLocally"]))
DTYPES = ["int64", "float64", "bool", "uint64"]

#   TODO unint8 ???


class TestPdarrayClass:

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_reshape(self):
        a = ak.arange(4)
        r = a.reshape((2, 2))
        assert r.shape == (2, 2)
        assert isinstance(r, ak.pdarray)

    def test_shape(self):
        a = ak.arange(4)
        np_a = np.arange(4)
        assert isinstance(a.shape, tuple)
        assert a.shape == np_a.shape

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_shape_multidim(self):
        a = ak.arange(4).reshape((2, 2))
        np_a = np.arange(4).reshape((2, 2))
        assert isinstance(a.shape, tuple)
        assert a.shape == np_a.shape

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flatten(self, size):
        a = ak.arange(size)
        ak_assert_equal(a.flatten(), a)

    @pytest.mark.skip_if_max_rank_less_than(3)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flatten(self, size):
        size = size - (size % 4)
        a = ak.arange(size)
        b = a.reshape((2, 2, size / 4))
        ak_assert_equal(b.flatten(), a)

    def test_prod(self):
        a = ak.arange(10) + 1

        assert ak.prod(a) == 3628800

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_prod_multidim(self):
        a = ak.ones((2, 3, 4))
        a = a + a

        assert ak.prod(a) == 2**24

        aProd0 = ak.prod(a, axis=0)
        assert aProd0.shape == (1, 3, 4)
        assert aProd0[0, 0, 0] == 2**2

        aProd02 = ak.prod(a, axis=(1, 2))
        assert aProd02.shape == (2, 1, 1)
        assert aProd02[0, 0, 0] == 2**12

    def test_sum(self):
        a = ak.ones(10)

        assert ak.sum(a) == 10

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_sum_multidim(self):
        a = ak.ones((2, 3, 4))

        assert ak.sum(a) == 24

        aSum0 = ak.sum(a, axis=0)
        assert aSum0.shape == (1, 3, 4)
        assert aSum0[0, 0, 0] == 2

        aSum02 = ak.sum(a, axis=(1, 2))
        assert aSum02.shape == (2, 1, 1)
        assert aSum02[0, 0, 0] == 12

    def test_max(self):
        a = ak.arange(10)
        assert ak.max(a) == 9

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_max_multidim(self):
        a = ak.array(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = 101

        assert ak.max(a) == 101

        aMax0 = ak.max(a, axis=0)
        assert aMax0.shape == (1, 7, 4)
        assert aMax0[0, 6, 2] == 101

        aMax02 = ak.max(a, axis=(0, 2))
        assert aMax02.shape == (1, 7, 1)
        assert aMax02[0, 6, 0] == 101

    def test_min(self):
        a = ak.arange(10) + 2
        assert ak.min(a) == 2

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_min_multidim(self):
        a = ak.array(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = -1

        assert ak.min(a) == -1

        aMin0 = ak.min(a, axis=0)
        assert aMin0.shape == (1, 7, 4)
        assert aMin0[0, 6, 2] == -1

        aMin02 = ak.min(a, axis=(0, 2))
        assert aMin02.shape == (1, 7, 1)
        assert aMin02[0, 6, 0] == -1

    def test_all(self):
        a = ak.ones(10, dtype=bool)
        assert ak.all(a)

        b = ak.zeros(10, dtype=bool)
        assert not ak.all(b)

        c = ak.array([True, True, False, True])
        assert not ak.all(c)

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_all_multidim(self):

        a = ak.ones((10, 10), dtype=bool)
        assert ak.all(a)

        b = ak.zeros((10, 10), dtype=bool)
        assert not ak.all(b)

        c = ak.array([True, True, False, True]).reshape((2, 2))
        assert not ak.all(c)

    def test_any(self):
        a = ak.ones(10, dtype=bool)
        assert ak.any(a)

        b = ak.zeros(10, dtype=bool)
        assert not ak.any(b)

        c = ak.array([True, True, False, True])
        assert ak.any(c)

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_any_multidim(self):
        a = ak.ones(10, dtype=bool)
        assert ak.any(a)

        b = ak.zeros(10, dtype=bool)
        assert not ak.any(b)

        c = ak.array([True, True, False, True])
        assert ak.any(c)

    def test_is_sorted(self):
        a = ak.arange(10)
        assert ak.is_sorted(a)

        b = a * -1
        assert not ak.is_sorted(b)

        c = ak.randint(0, 10, 10)
        assert not ak.is_sorted(c)

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_is_sorted_multidim(self):
        a = ak.array(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        assert not ak.is_sorted(a)

        x = ak.arange(10).reshape((2, 5))
        assert ak.is_sorted(x)

    def test_is_locally_sorted(self):
        from arkouda.pdarrayclass import is_locally_sorted

        a = ak.arange(10)
        assert is_locally_sorted(a)

        b = a * -1
        assert not is_locally_sorted(b)

        c = ak.randint(0, 10, 10)
        assert not is_locally_sorted(c)

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_is_locally_sorted_multidim(self):
        from arkouda.pdarrayclass import is_locally_sorted

        a = ak.array(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        assert not is_locally_sorted(a)

        x = ak.arange(10).reshape((2, 5))
        assert is_locally_sorted(x)

    def assert_ops_match(self, op: str, pda: ak.pdarray):
        ak_op = getattr(arkouda.pdarrayclass, op)
        np_op = getattr(numpy, op)
        nda = pda.to_ndarray()
        assert ak_op(pda) == np_op(nda)

    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_reductions_match_numpy_1D_arange(self, op, size, dtype):
        size = min(size, 1000) if op == "prod" else size
        pda = ak.arange(size, dtype=dtype)
        self.assert_ops_match(op, pda)

    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_reductions_match_numpy_1D_ones(self, op, size, dtype):
        size = min(size, 1000) if op == "prod" else size
        pda = ak.ones(size, dtype=dtype)
        self.assert_ops_match(op, pda)


    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_reductions_match_numpy_1D_zeros(self, op, size, dtype):
        size = min(size, 1000) if op == "prod" else size
        pda = ak.zeros(size, dtype=dtype)
        self.assert_ops_match(op, pda)

    @pytest.mark.parametrize("op", REDUCTION_OPS)
    def test_reductions_match_numpy_1D_TF(self, op):
        pda = ak.array([True, True, False, True])
        self.assert_ops_match(op, pda)

    @pytest.mark.skip_if_max_rank_less_than(3)
    @pytest.mark.parametrize("op", REDUCTION_OPS)
    def test_reductions_match_numpy_1D_TF(self, op):
        pda = ak.array([True, True, False, True])
        self.assert_ops_match(op, pda)

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_any_multidim(self):
        a = ak.ones(10, dtype=bool)
        assert ak.any(a)

        b = ak.zeros(10, dtype=bool)
        assert not ak.any(b)

        c = ak.array([True, True, False, True]).reshape((2,2))
        assert ak.any(c)
