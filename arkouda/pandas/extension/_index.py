import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.core.indexes.base import Index

import arkouda as ak

from ._arkouda_array import ArkoudaArray


class ArkoudaIndex(pd.Index):
    _typ = "arkoudaindex"
    _data: ArkoudaArray

    def __new__(cls, data, dtype=None, copy=False, name=None):
        if not isinstance(data, ArkoudaArray):
            data = ArkoudaArray(ak.array(data))
        return pd.Index.__new__(cls, data, dtype=dtype, copy=copy, name=name)
