from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Literal, NoReturn, cast

import numpy as np
import pandas as pd
from pandas._libs.missing import is_matching_na
from pandas.api.types import is_bool, is_number
from pandas.io.formats.printing import pprint_thing

from arkouda import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Strings,
    all,
    argsort,
    array,
    pdarray,
    sort,
    sum,
)
from arkouda.numpy import nan
from arkouda.util import is_float_dtype, is_integer_dtype, is_numeric, is_numeric_dtype

# from pandas.core.dtypes.common import (
#     is_bool,
#     is_float_dtype,
#     is_integer_dtype,
#     is_number,
#     is_numeric_dtype,
#     needs_i8_conversion,
# )
# from pandas.core.dtypes.dtypes import (
#     CategoricalDtype,
#     DatetimeTZDtype,
#     ExtensionDtype,
#     NumpyEADtype,
# )

# import pandas as pd
# from pandas import (
#     Categorical,
#     DataFrame,
#     DatetimeIndex,
#     Index,
#     IntervalDtype,
#     IntervalIndex,
#     MultiIndex,
#     PeriodIndex,
#     RangeIndex,
#     Series,
#     TimedeltaIndex,
# )
# from pandas.core.arrays import (
#     DatetimeArray,
#     ExtensionArray,
#     IntervalArray,
#     PeriodArray,
#     TimedeltaArray,
# )
#
# from pandas.core.arrays.string_ import StringDtype
# from pandas.core.indexes.api import safe_sort_index
#
# from pandas.io.formats.printing import pprint_thing
#
# if TYPE_CHECKING:
#     from pandas._typing import DtypeObj

DEBUG = True


def assert_almost_equal(
    left,
    right,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    **kwargs,
) -> None:
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    Parameters
    ----------
    left : object
    right : object
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    """
    if isinstance(left, Index):
        assert_index_equal(
            left,
            right,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, Series):
        assert_series_equal(
            left,
            right,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, DataFrame):
        assert_frame_equal(
            left,
            right,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    else:
        # Other sequences.

        if is_number(left) and is_number(right):
            # Do not compare numeric classes, like np.float64 and float.
            pass
        elif is_bool(left) and is_bool(right):
            # Do not compare bool classes, like np.bool_ and bool.
            pass
        else:
            if isinstance(left, pdarray) or isinstance(right, pdarray):
                obj = "pdarray"
            else:
                obj = "Input"
            assert_class_equal(left, right, obj=obj)

        # if we have "equiv", this becomes True
        assert np.allclose(left.to_ndarray(), right.to_ndarray(), rtol=rtol, atol=atol, equal_nan=True)


def _check_isinstance(left, right, cls) -> None:
    """
    Helper method for our assert_* methods that ensures that
    the two objects being compared have the right type before
    proceeding with the comparison.

    Parameters
    ----------
    left : The first object being compared.
    right : The second object being compared.
    cls : The class type to check against.

    Raises
    ------
    AssertionError : Either `left` or `right` is not an instance of `cls`.
    """
    cls_name = cls.__name__

    if not isinstance(left, cls):
        raise AssertionError(f"{cls_name} Expected type {cls}, found {type(left)} instead")
    if not isinstance(right, cls):
        raise AssertionError(f"{cls_name} Expected type {cls}, found {type(right)} instead")


def assert_dict_equal(left, right, compare_keys: bool = True) -> None:
    _check_isinstance(left, right, dict)

    left_keys = frozenset(left.keys())
    right_keys = frozenset(right.keys())

    if compare_keys:
        assert left_keys == right_keys

    for k in left_keys:
        assert_almost_equal(left[k], right[k])

    return True


def assert_index_equal(
    left: Index,
    right: Index,
    exact: bool = True,
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "Index",
) -> None:
    """
    Check that left and right Index are equal.

    Parameters
    ----------
    left : Index
    right : Index
    exact : True
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_names : bool, default True
        Whether to check the names attribute.
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Index'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Index([1, 2, 3])
    >>> b = pd.Index([1, 2, 3])
    >>> tm.assert_index_equal(a, b)
    """
    __tracebackhide__ = not DEBUG

    def _check_types(left, right, obj: str = "Index") -> None:
        if not exact:
            return

        assert_class_equal(left, right, exact=exact, obj=obj)
        assert_attr_equal("inferred_type", left, right, obj=obj)

        # Skip exact dtype checking when `check_categorical` is False
        if isinstance(left.dtype, Categorical) and isinstance(right.dtype, Categorical):
            if check_categorical:
                assert_attr_equal("dtype", left, right, obj=obj)
                assert_index_equal(left.categories, right.categories, exact=exact)
            return

        assert_attr_equal("dtype", left, right, obj=obj)

    # instance validation
    _check_isinstance(left, right, Index)

    # class / dtype comparison
    _check_types(left, right, obj=obj)

    # level comparison
    if left.nlevels != right.nlevels:
        msg1 = f"{obj} levels are different"
        msg2 = f"{left.nlevels}, {left}"
        msg3 = f"{right.nlevels}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{obj} length are different"
        msg2 = f"{len(left)}, {left}"
        msg3 = f"{len(right)}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # If order doesn't matter then sort the index entries
    if not check_order:
        left = left[left.argsort()]  # safe_sort_index(left)
        right = right[right.argsort()]  # safe_sort_index(right)

    # MultiIndex special comparison for little-friendly error messages
    if isinstance(left, MultiIndex):
        right = cast(MultiIndex, right)

        for level in range(left.nlevels):
            lobj = f"MultiIndex level [{level}]"
            try:
                # try comparison on levels/codes to avoid densifying MultiIndex
                assert_index_equal(
                    left.levels[level],
                    right.levels[level],
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
                assert_arkouda_array_equal(left.codes[level], right.codes[level])
            except AssertionError:
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)

                assert_index_equal(
                    llevel,
                    rlevel,
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
            # get_level_values may change dtype
            _check_types(left.levels[level], right.levels[level], obj=obj)

    # skip exact index checking when `check_categorical` is False
    # differed from pandas due to unintuitive pandas behavior.
    elif check_exact is True or not is_numeric(left) or not is_numeric(right):
        if not left.equals(right):
            if isinstance(left, list) and isinstance(right, list):
                mismatch = np.array(left) != np.array(right)
            else:
                mismatch = left != right

            diff = sum(mismatch.astype(int)) * 100.0 / len(left)
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right)
    else:
        # @TODO Use new ak.allclose function
        assert_almost_equal(left.values, right.values, rtol=rtol, atol=atol, equal_nan=True)

    # metadata comparison
    if check_names:
        assert_attr_equal("names", left, right, obj=obj)

    if check_categorical:
        if isinstance(left.dtype, Categorical) or isinstance(right.dtype, Categorical):
            assert_categorical_equal(left.values, right.values, obj=f"{obj} category")


def assert_class_equal(left, right, exact: bool = True, obj: str = "Input") -> None:
    """
    Checks classes are equal.
    """
    # __tracebackhide__ = True

    def repr_class(x):
        if isinstance(x, Index):
            # return Index as it is to include values in the error message
            return x

        return type(x).__name__

    if type(left) == type(right):
        return

    msg = f"{obj} classes are different"
    raise_assert_detail(obj, msg, repr_class(left), repr_class(right))


def assert_attr_equal(attr: str, left, right, obj: str = "Attributes") -> None:
    """
    Check attributes are equal. Both objects must have attribute.

    Parameters
    ----------
    attr : str
        Attribute name being compared.
    left : object
    right : object
    obj : str, default 'Attributes'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    # __tracebackhide__ = True

    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)

    if left_attr is right_attr or is_matching_na(left_attr, right_attr):
        # e.g. both np.nan, both NaT, both pd.NA, ...
        return None

    try:
        result = left_attr == right_attr
    except TypeError:
        result = False
    if (left_attr is nan) ^ (right_attr is nan):
        result = False
    elif not isinstance(result, bool):
        result = result.all()

    if not result:
        msg = f'Attribute "{attr}" are different'
        raise_assert_detail(obj, msg, left_attr, right_attr)
    return None


def assert_is_sorted(seq) -> None:
    """Assert that the sequence is sorted."""
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    # sorting does not change precisions
    assert_arkouda_array_equal(seq, sort(array(seq)))


def assert_categorical_equal(
    left,
    right,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = "Categorical",
) -> None:
    """
    Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
    right : Categorical
    check_dtype : bool, default True
        Check that integer dtype of the codes are the same.
    check_category_order : bool, default True
        Whether the order of the categories should be compared, which
        implies identical integer codes.  If False, only the resulting
        values are compared.  The ordered attribute is
        checked regardless.
    obj : str, default 'Categorical'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    """
    _check_isinstance(left, right, Categorical)

    exact: bool | str

    exact = True

    if check_category_order is True:
        assert_arkouda_array_equal(
            left.categories, right.categories, check_dtype=check_dtype, obj=f"{obj}.categories"
        )
        assert_arkouda_array_equal(left.codes, right.codes, check_dtype=check_dtype, obj=f"{obj}.codes")
    else:
        try:
            lc = left.sort().categories
            rc = right.sort().categories
        except TypeError:
            # e.g. '<' not supported between instances of 'int' and 'str'
            lc, rc = left.categories, right.categories
        assert_arkouda_array_equal(lc, rc, obj=f"{obj}.categories")
        left_sorted = left[argsort(left)]
        right_sorted = right[argsort(right)]
        assert_arkouda_array_equal(
            left.categories[left.codes],
            right.categories[right.codes],
            obj=f"{obj}.values",
        )


def raise_assert_detail(
    obj, message, left, right, diff=None, first_diff=None, index_values=None
) -> NoReturn:
    # __tracebackhide__ = True

    msg = f"""{obj} are different

{message}"""

    if isinstance(index_values, Index):
        index_values = index_values.values.to_ndarray()

    if isinstance(index_values, pdarray):
        index_values = index_values.to_ndarray()

    if isinstance(index_values, np.ndarray):
        msg += f"\n[index]: {pprint_thing(index_values)}"

    if isinstance(left, pdarray):
        left = left.to_ndarray()
    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif isinstance(left, (Categorical, Strings)):
        left = repr(left)

    if isinstance(right, pdarray):
        right = right.to_ndarray()
    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif isinstance(right, (Categorical, Strings)):
        right = repr(right)

    msg += f"""
[left]:  {left}
[right]: {right}"""

    if diff is not None:
        msg += f"\n[diff]: {diff}"

    if first_diff is not None:
        msg += f"\n{first_diff}"

    raise AssertionError(msg)


def assert_arkouda_pdarray_equal(
    left,
    right,
    check_dtype: bool = True,
    err_msg=None,
    check_same=None,
    obj: str = "pdarray",
    index_values=None,
) -> None:
    """
    Check that 'ak.pdarray' is equivalent.

    Parameters
    ----------
    left, right : arkouda.pdarray
        The two arrays to be compared.
    check_dtype : bool, default True
        Check dtype if both a and b are ak.pdarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | arkouda.pdarray, default None
        optional index (shared by both left and right), used in output.
    """
    # __tracebackhide__ = True

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an ak.pdarray
    _check_isinstance(left, right, pdarray)

    def _get_base(obj):
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{repr(left_base)} is not {repr(right_base)}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{repr(left_base)} is {repr(right_base)}")

    def _raise(left: pdarray, right: pdarray, err_msg):
        if err_msg is None:
            if left.shape != right.shape:
                raise_assert_detail(obj, f"{obj} shapes are different", left.shape, right.shape)

            diff = sum(left != right)

            diff = diff * 100.0 / left.size
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right, index_values=index_values)

        raise AssertionError(err_msg)

    # compare shape and values
    # @TODO use ak.allclose

    from arkouda import all as akall
    from arkouda.dtypes import bigint, dtype

    if isinstance(left, pdarray) and isinstance(right, pdarray) and left.dtype == dtype(bigint):
        if not akall(left == right):
            _raise(left, right, err_msg)
    elif not np.allclose(left.to_ndarray(), right.to_ndarray(), atol=0, equal_nan=True):
        _raise(left, right, err_msg)

    if check_dtype:
        if isinstance(left, pdarray) and isinstance(right, pdarray):
            assert_attr_equal("dtype", left, right, obj=obj)


def assert_arkouda_strings_equal(
    left,
    right,
    err_msg=None,
    check_same=None,
    obj: str = "Strings",
    index_values=None,
) -> None:
    """
    Check that 'ak.pdarray' is equivalent.

    Parameters
    ----------
    left, right : arkouda.Strings
        The two Strings to be compared.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | arkouda.pdarray, default None
        optional index (shared by both left and right), used in output.
    """
    # __tracebackhide__ = True

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an ak.pdarray
    _check_isinstance(left, right, Strings)

    def _get_base(obj):
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{repr(left_base)} is not {repr(right_base)}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{repr(left_base)} is {repr(right_base)}")

    def _raise(left: Strings, right: Strings, err_msg):
        if err_msg is None:
            diff = sum(left != right)
            diff = diff * 100.0 / left.size
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right, index_values=index_values)

        raise AssertionError(err_msg)

    if left.shape != right.shape:
        raise_assert_detail(obj, f"{obj} shapes are different", left.shape, right.shape)

    if not sum(left != right) == 0:
        _raise(left, right, err_msg)


def assert_arkouda_array_equal(
    left: pdarray | Strings,
    right: pdarray | Strings,
    check_dtype: bool = True,
    err_msg=None,
    check_same=None,
    obj: str = "pdarray",
    index_values=None,
) -> None:
    """
    Check that 'ak.pdarray' is equivalent.

    Parameters
    ----------
    left, right : arkouda.pdarray or arkouda.Strings
        The two arrays to be compared.
    check_dtype : bool, default True
        Check dtype if both a and b are ak.pdarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | arkouda.pdarray, default None
        optional index (shared by both left and right), used in output.
    """
    if isinstance(left, Strings):
        print("case1")
        print(left)
        assert_arkouda_strings_equal(
            left, right, err_msg=err_msg, check_same=check_same, obj=obj, index_values=index_values
        )
    elif isinstance(left, Categorical):
        assert_arkouda_array_equal(
            left.categories[left.codes],
            right.categories[right.codes],
            check_dtype=check_dtype,
            err_msg=err_msg,
            check_same=check_same,
            obj=obj,
            index_values=index_values,
        )
    else:
        assert_arkouda_pdarray_equal(
            left,
            right,
            check_dtype=check_dtype,
            err_msg=err_msg,
            check_same=check_same,
            obj=obj,
            index_values=index_values,
        )


# This could be refactored to use the NDFrame.equals method
def assert_series_equal(
    left,
    right,
    check_dtype: bool = True,
    check_index_type: bool = True,
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_category_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "Series",
    *,
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    """
    Check that left and right Series are equal.

    Parameters
    ----------
    left : Series
    right : Series
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool, default True
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
         Whether to check the Series class is identical.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    check_index : bool, default True
        Whether to check index equivalence. If False, then compare only values.

        .. versionadded:: 1.3.0
    check_like : bool, default False
        If True, ignore the order of the index. Must be False if check_index is False.
        Note: same labels must be with the same data.

        .. versionadded:: 1.5.0

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b = pd.Series([1, 2, 3, 4])
    >>> tm.assert_series_equal(a, b)
    """
    # __tracebackhide__ = True
    check_exact_index = check_exact  ## TODO Remove

    if not check_index and check_like:
        raise ValueError("check_like must be False if check_index is False")

    # instance validation
    _check_isinstance(left, right, Series)

    if check_series_type:
        assert_class_equal(left, right, obj=obj)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{len(left)}, {left.index}"
        msg2 = f"{len(right)}, {right.index}"
        raise_assert_detail(obj, "Series length are different", msg1, msg2)

    if check_index:
        assert_index_equal(
            left.index,
            right.index,
            exact=check_index_type,
            check_names=check_names,
            check_exact=check_exact_index,
            check_categorical=check_categorical,
            check_order=not check_like,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.index",
        )

    if check_like:
        left = left.reindex_like(right)

    if check_dtype:
        # We want to skip exact dtype checking when `check_categorical`
        # is False. We'll still raise if only one is a `Categorical`,
        # regardless of `check_categorical`
        if (
            isinstance(left.dtype, Categorical)
            and isinstance(right.dtype, Categorical)
            and not check_categorical
        ):
            pass
        else:
            assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")

    if check_exact or not is_numeric(left.values) or not is_numeric(right.values):
        lv, rv = left.values, right.values
        assert_arkouda_array_equal(
            lv,
            rv,
            check_dtype=check_dtype,
            obj=str(obj),
            index_values=left.index,
        )
    else:
        assert_almost_equal(
            left.values,
            right.values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("name", left, right, obj=obj)

    if check_categorical is True:
        if isinstance(left.values, Categorical) or isinstance(right.values, Categorical):
            assert_categorical_equal(
                left.values,
                right.values,
                obj=f"{obj} category",
                check_category_order=check_category_order,
                check_dtype=check_dtype,
            )


# This could be refactored to use the NDFrame.equals method
def assert_frame_equal(
    left,
    right,
    check_dtype: bool = True,
    check_index_type: bool = True,
    check_column_type: bool = True,
    check_frame_type: bool = True,
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_like: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "DataFrame",
) -> None:
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. It is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool, default = True
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.
    DataFrame.equals : Check DataFrame equality.

    Examples
    --------
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from pandas.testing import assert_frame_equal
    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

    df1 equals itself.

    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.

    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    ...
    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.

    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    # __tracebackhide__ = True
    _rtol = rtol
    _atol = atol
    _check_exact = check_exact

    # instance validation
    _check_isinstance(left, right, DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))
        assert_class_equal(left, right, obj=obj)

    # shape comparison
    if left.shape != right.shape:
        raise_assert_detail(obj, f"{obj} shape mismatch", f"{repr(left.shape)}", f"{repr(right.shape)}")

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.index",
    )

    # column comparison
    assert_index_equal(
        left.columns,
        right.columns,
        exact=check_column_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.columns",
    )

    if check_like:
        # left = left.reindex_like(right)
        left = left[right.index.values]

    for col in left.columns.values:
        # We have already checked that columns match, so we can do
        #  fast location-based lookups
        lcol = left[col]
        rcol = right[col]
        if not isinstance(lcol, Series):
            lcol = Series(lcol)
        if not isinstance(rcol, Series):
            rcol = Series(rcol)

        # use check_index=False, because we do not want to run
        # assert_index_equal for each column,
        # as we already checked it for the whole dataframe before.
        assert_series_equal(
            lcol,
            rcol,
            check_dtype=check_dtype,
            check_index_type=check_index_type,
            check_exact=check_exact,
            check_names=check_names,
            check_categorical=check_categorical,
            obj=f'{obj}.(column name="{col}")',
            rtol=rtol,
            atol=atol,
            check_index=False,
        )


def assert_equal(left, right, **kwargs) -> None:
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.
    """
    # __tracebackhide__ = True

    if isinstance(left, Index):
        assert_index_equal(left, right, **kwargs)
    elif isinstance(left, Series):
        assert_series_equal(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        assert_arkouda_array_equal(left, right, **kwargs)
    elif isinstance(left, str):
        assert kwargs == {}
        assert left == right
    else:
        assert kwargs == {}
        assert_almost_equal(left, right)


def assert_contains_all(iterable, dic) -> None:
    for k in iterable:
        assert k in dic, f"Did not contain item: {repr(k)}"


def assert_copy(iter1, iter2, **eql_kwargs) -> None:
    """
    iter1, iter2: iterables that produce elements
    comparable with assert_almost_equal

    Checks that the elements are equal, but not
    the same object. (Does not check that items
    in sequences are also not the same object)
    """
    for elem1, elem2 in zip(iter1, iter2):
        assert_almost_equal(elem1, elem2, **eql_kwargs)
        msg = (
            f"Expected object {repr(type(elem1))} and object {repr(type(elem2))} to be "
            "different objects, but they were the same object."
        )
        assert elem1 is not elem2, msg


def assert_indexing_slices_equivalent(ser: Series, l_slc: slice, i_slc: slice) -> None:
    """
    Check that ser.iloc[i_slc] matches ser.loc[l_slc] and, if applicable,
    ser[l_slc].
    """
    expected = ser.iloc[i_slc]

    assert_series_equal(ser.loc[l_slc], expected)

    if not is_integer_dtype(ser.index):
        # For integer indices, .loc and plain getitem are position-based.
        assert_series_equal(ser[l_slc], expected)
