import pandas as pd
import pytest
from numpy import dtype as npdtype

import arkouda as ak
from arkouda.dtypes import dtype
from arkouda.index import Index
from arkouda.pdarrayclass import pdarray


class TestIndex:
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_index_creation(self, size):
        idx = ak.Index(ak.arange(size))

        assert isinstance(idx, ak.Index)
        assert idx.size == size
        assert idx.to_list() == list(range(size))

    def test_index_creation_lists(self):
        i = ak.Index([1, 2, 3])
        assert isinstance(i.values, pdarray)

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert isinstance(i2.values, list)
        assert i2.dtype == dtype("int64")

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert isinstance(i3.values, list)
        assert i3.dtype == dtype("<U")

        with pytest.raises(ValueError):
            i4 = ak.Index([1, 2, 3], allow_list=True, max_list_size=2)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multiindex_creation(self, size):
        # test list generation
        idx = ak.MultiIndex([ak.arange(size), ak.arange(size)])
        assert isinstance(idx, ak.MultiIndex)
        assert idx.nlevels == 2
        assert idx.size == size

        # test tuple generation
        idx = ak.MultiIndex((ak.arange(size), ak.arange(size)))
        assert isinstance(idx, ak.MultiIndex)
        assert idx.nlevels == 2
        assert idx.size == size

        with pytest.raises(TypeError):
            idx = ak.MultiIndex(ak.arange(size))

        with pytest.raises(ValueError):
            idx = ak.MultiIndex([ak.arange(size), ak.arange(size - 1)])

    def test_name_names(self):
        i = ak.Index([1, 2, 3], name="test")
        assert i.name == "test"
        assert i.names == ["test"]

        size = 10
        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1], names=["test", "test2"])
        assert m.names == ["test", "test2"]

    def test_nlevels(self):
        i = ak.Index([1, 2, 3], name="test")
        assert i.nlevels == 1

        size = 10
        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1])
        assert m.nlevels == 2

    def test_ndim(self):
        i = ak.Index([1, 2, 3], name="test")
        assert i.ndim == 1

        size = 10
        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1])
        assert m.ndim == 1

    def test_dtypes(self):
        size = 10
        i = ak.Index(ak.arange(size, dtype="float64"))
        assert i.dtype == dtype("float64")

        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1])
        assert m.dtype == npdtype("O")

    def test_inferred_type(self):
        i = ak.Index([1, 2, 3])
        assert i.inferred_type == "integer"

        i2 = ak.Index([1.0, 2, 3])
        assert i2.inferred_type == "floating"

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i3.inferred_type == "string"

        from arkouda.categorical import Categorical

        i4 = ak.Index(Categorical(ak.array(["a", "b", "c"])))
        assert i4.inferred_type == "categorical"

        size = 10
        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1], names=["test", "test2"])
        assert m.inferred_type == "mixed"

    @staticmethod
    def assert_equal(pda1, pda2):
        from arkouda import sum as aksum

        assert pda1.size == pda2.size
        assert aksum(pda1 != pda2) == 0

    def test_get_item(self):
        i = ak.Index([1, 2, 3])
        assert i[2] == 3
        assert isinstance(i[[0, 1]], Index)
        assert i[[0, 1]].equals(Index([1, 2]))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert i2[2] == 3
        assert i2[[0, 1]].equals(Index([1, 2], allow_list=True))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i3[2] == "c"
        assert i3[[0, 1]].equals(Index(["a", "b"], allow_list=True))

    def test_eq(self):
        i = ak.Index([1, 2, 3])
        i_cpy = ak.Index([1, 2, 3])
        self.assert_equal(i == i_cpy, ak.array([True, True, True]))
        self.assert_equal(i != i_cpy, ak.array([False, False, False]))
        assert i.equals(i_cpy)

        i2 = ak.Index([1, 2, 3], allow_list=True)
        i2_cpy = ak.Index([1, 2, 3], allow_list=True)
        self.assert_equal(i2 == i2_cpy, ak.array([True, True, True]))
        self.assert_equal(i2 != i2_cpy, ak.array([False, False, False]))
        assert i2.equals(i2_cpy)

        self.assert_equal(i == i2, ak.array([True, True, True]))
        self.assert_equal(i != i2, ak.array([False, False, False]))
        assert i.equals(i2)

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        i3_cpy = ak.Index(["a", "b", "c"], allow_list=True)
        self.assert_equal(i3 == i3_cpy, ak.array([True, True, True]))
        self.assert_equal(i3 != i3_cpy, ak.array([False, False, False]))
        assert i3.equals(i3_cpy)

        i4 = ak.Index(["a", "b", "c"], allow_list=False)
        i4_cpy = ak.Index(["a", "b", "c"], allow_list=False)
        self.assert_equal(i4 == i4_cpy, ak.array([True, True, True]))
        self.assert_equal(i4 != i4_cpy, ak.array([False, False, False]))
        assert i3.equals(i3_cpy)

        i5 = ak.Index(["a", "x", "c"], allow_list=True)
        self.assert_equal(i3 == i5, ak.array([True, False, True]))
        self.assert_equal(i3 != i5, ak.array([False, True, False]))
        assert not i3.equals(i5)

        i6 = ak.Index(["a", "b", "c", "d"], allow_list=True)
        assert not i5.equals(i6)

        with pytest.raises(ValueError):
            i5 == i6

        with pytest.raises(ValueError):
            i5 != i6

        with pytest.raises(TypeError):
            i.equals("string")

    def test_multiindex_equals(self):
        size = 10
        arrays = [ak.array([1, 1, 2, 2]), ak.array(["red", "blue", "red", "blue"])]
        m = ak.MultiIndex(arrays, names=["numbers", "colors"])
        assert m.equals(m)

        arrays2 = [ak.array([1, 1, 2, 2]), ak.array(["red", "blue", "red", "blue"])]
        m2 = ak.MultiIndex(arrays2, names=["numbers2", "colors2"])
        assert m.equals(m2)

        arrays3 = [
            ak.array([1, 1, 2, 2]),
            ak.array(["red", "blue", "red", "blue"]),
            ak.array([1, 1, 2, 2]),
        ]
        m3 = ak.MultiIndex(arrays3, names=["numbers", "colors2", "numbers3"])
        assert not m.equals(m3)

        arrays4 = [ak.array([1, 1, 2, 2]), ak.array(["red", "blue", "red", "green"])]
        m4 = ak.MultiIndex(arrays4, names=["numbers2", "colors2"])
        assert not m.equals(m4)

        m5 = ak.MultiIndex([ak.arange(size)])
        i = ak.Index(ak.arange(size))
        assert not m5.equals(i)
        assert not i.equals(m5)

    def test_equal_levels(self):
        m = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"])],
            names=["col1", "col2", "col3"],
        )

        m2 = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"])],
            names=["A", "B", "C"],
        )

        assert m.equal_levels(m2)

        m3 = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"]), 2 * ak.arange(3)],
            names=["col1", "col2", "col3"],
        )

        assert not m.equal_levels(m3)

        m4 = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * 2, ak.array(["a", 'b","c', "d"])],
            names=["col1", "col2", "col3"],
        )

        assert not m.equal_levels(m4)

    def test_get_level_values(self):
        m = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"])],
            names=["col1", "col2", "col3"],
        )

        i1 = Index(ak.arange(3), name="col1")
        self.assert_equal(m.get_level_values(0), i1)
        self.assert_equal(m.get_level_values("col1"), i1)

        i2 = Index(ak.arange(3) * -1, name="col2")
        self.assert_equal(m.get_level_values(1), i2)
        self.assert_equal(m.get_level_values("col2"), i2)

        i3 = Index(ak.array(["a", 'b","c', "d"]), name="col3")
        self.assert_equal(m.get_level_values(2), i3)
        self.assert_equal(m.get_level_values("col3"), i3)

        with pytest.raises(ValueError):
            m.get_level_values("col4")

        #   Test when names=None
        m2 = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"])],
        )
        i4 = Index(ak.arange(3))
        self.assert_equal(m2.get_level_values(0), i4)

        with pytest.raises(RuntimeError):
            m2.get_level_values("col")

        with pytest.raises(ValueError):
            m2.get_level_values(m2.nlevels)

        with pytest.raises(ValueError):
            m2.get_level_values(-1 * m2.nlevels)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_memory_usage(self, size):
        from arkouda.dtypes import BigInt
        from arkouda.index import Index, MultiIndex

        idx = Index(ak.cast(ak.array([1, 2, 3]), dt="bigint"))
        assert idx.memory_usage() == 3 * BigInt.itemsize

        idx = Index(ak.cast(ak.arange(size), dt="int64"))
        assert idx.memory_usage(unit="GB") == size * ak.dtypes.int64.itemsize / (1024 * 1024 * 1024)
        assert idx.memory_usage(unit="MB") == size * ak.dtypes.int64.itemsize / (1024 * 1024)
        assert idx.memory_usage(unit="KB") == size * ak.dtypes.int64.itemsize / 1024
        assert idx.memory_usage(unit="B") == size * ak.dtypes.int64.itemsize

        midx = MultiIndex([ak.cast(ak.arange(size), dt="int64"), ak.cast(ak.arange(size), dt="int64")])
        assert midx.memory_usage(unit="GB") == 2 * size * ak.dtypes.int64.itemsize / (1024 * 1024 * 1024)

        assert midx.memory_usage(unit="MB") == 2 * size * ak.dtypes.int64.itemsize / (1024 * 1024)
        assert midx.memory_usage(unit="KB") == 2 * size * ak.dtypes.int64.itemsize / 1024
        assert midx.memory_usage(unit="B") == 2 * size * ak.dtypes.int64.itemsize

    def test_is_unique(self):
        i = ak.Index(ak.array([0, 1, 2]))
        assert i.is_unique

        i = ak.Index(ak.array([0, 1, 1]))
        assert not i.is_unique

        i = ak.Index([0, 1, 2], allow_list=True)
        assert i.is_unique

        i = ak.Index([0, 1, 1], allow_list=True)
        assert not i.is_unique

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_factory(self, size):
        idx = ak.Index.factory(ak.arange(size))
        assert isinstance(idx, ak.Index)

        idx = ak.Index.factory([ak.arange(size), ak.arange(size)])
        assert isinstance(idx, ak.MultiIndex)

    def test_argsort(self):
        idx = ak.Index.factory(ak.arange(5))
        i = idx.argsort(False)
        assert i.to_list() == [4, 3, 2, 1, 0]

        idx = ak.Index(ak.array([1, 0, 4, 2, 5, 3]))
        i = idx.argsort()
        # values should be the indexes in the array of idx
        assert i.to_list() == [1, 0, 3, 5, 2, 4]

        i = ak.Index([1, 2, 3])
        assert i.argsort(ascending=True).to_list() == [0, 1, 2]
        assert i.argsort(ascending=False).to_list() == [2, 1, 0]

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert i2.argsort(ascending=True) == [0, 1, 2]
        assert i2.argsort(ascending=False) == [2, 1, 0]

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i3.argsort(ascending=True) == [0, 1, 2]
        assert i3.argsort(ascending=False) == [2, 1, 0]

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        assert i4.argsort(ascending=True).to_list() == [0, 1, 2]
        assert i4.argsort(ascending=False).to_list() == [2, 1, 0]

    def test_map(self):
        idx = ak.Index(ak.array([2, 3, 2, 3, 4]))

        result = idx.map({4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        assert result.values.to_list() == [30.0, 5.0, 30.0, 5.0, 25.0]

    def test_concat(self):
        idx_1 = ak.Index.factory(ak.arange(5))

        idx_2 = ak.Index(ak.array([2, 4, 1, 3, 0]))

        idx_full = idx_1.concat(idx_2)
        assert idx_full.to_list() == [0, 1, 2, 3, 4, 2, 4, 1, 3, 0]

        i = ak.Index([1, 2, 3], allow_list=True)
        i2 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i.concat(i2).to_list() == ["1", "2", "3", "a", "b", "c"]

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_lookup(self, size):
        idx = ak.Index.factory(ak.arange(size))
        lk = idx.lookup(ak.array([0, size - 1]))

        assert lk.to_list() == [i in [0, size - 1] for i in range(size)]

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_argsort(self, size):
        idx = ak.Index.factory([ak.arange(size), ak.arange(size)])
        s = idx.argsort(False)
        assert s.to_list() == list(reversed(range(size)))

        s = idx.argsort()
        assert s.to_list() == list(range(size))

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_concat(self, size):
        idx = ak.Index.factory([ak.arange(size), ak.arange(size)])
        idx_2 = ak.Index.factory(ak.arange(size) + 0.1)
        with pytest.raises(TypeError):
            idx.concat(idx_2)

        idx_2 = ak.Index.factory([ak.arange(size), ak.arange(size)])
        idx_full = idx.concat(idx_2)
        assert idx_full.to_pandas().tolist() == [(i, i) for i in range(size)] * 2

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_lookup(self, size):
        idx = ak.Index.factory([ak.arange(size), ak.arange(size)])
        truth = [0, 3, 2]
        lk = ak.array(truth)
        result = idx.lookup([lk, lk])

        assert result.to_list() == [i in truth for i in range(size)]

    def test_to_pandas(self):
        i = ak.Index([1, 2, 3])
        assert i.to_pandas().equals(pd.Index([1, 2, 3]))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert i2.to_pandas().equals(pd.Index([1, 2, 3]))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i3.to_pandas().equals(pd.Index(["a", "b", "c"]))

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        assert i4.to_pandas().equals(pd.Index(["a", "b", "c"]))

    def test_to_ndarray(self):
        from numpy import array as ndarray
        from numpy import array_equal

        i = ak.Index([1, 2, 3])
        assert array_equal(i.to_ndarray(), ndarray([1, 2, 3]))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert array_equal(i2.to_ndarray(), ndarray([1, 2, 3]))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert array_equal(i3.to_ndarray(), ndarray(["a", "b", "c"]))

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        assert array_equal(i4.to_ndarray(), ndarray(["a", "b", "c"]))

    def test_to_list(self):
        i = ak.Index([1, 2, 3])
        assert i.to_list() == [1, 2, 3]

        i2 = ak.Index([1, 2, 3], allow_list=True)
        assert i2.to_list() == [1, 2, 3]

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        assert i3.to_list() == ["a", "b", "c"]

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        assert i4.to_list() == ["a", "b", "c"]

    def test_register_list_values(self):
        i = ak.Index([1, 2, 3], allow_list=True)
        with pytest.raises(TypeError):
            i.register("test")
