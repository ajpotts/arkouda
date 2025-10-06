from pandas import DataFrame as pd_DataFrame
from pandas.api.extensions import register_dataframe_accessor

from arkouda.numpy.pdarraycreation import array as ak_array

from ._arkouda_array import ArkoudaArray


@register_dataframe_accessor("ak")
class ArkoudaAccessor:
    """
    Bare-bones Arkouda DataFrame accessor.

    Allows `df.ak` access to Arkouda-backed operations.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def to_ak(self):
        """
        Convert the pandas DataFrame to an Arkouda DataFrame.
        """
        cols = {}
        for name, col in self._obj.items():
            # simplistic conversion; assumes numeric or string
            # if ak.is_registered(col.name):
            #     cols[name] = ak_attach(col.name)
            # else:
            cols[name] = ArkoudaArray(ak_array(col.values))
        return pd_DataFrame(cols)

    def from_ak(self, akdf):
        """
        Convert an Arkouda DataFrame back to pandas.
        """
        import pandas as pd

        cols = {name: akdf[name].to_ndarray() for name in akdf.columns}
        return pd.DataFrame(cols)

    def info(self):
        """
        Simple Arkouda-like summary of the underlying pandas DataFrame.
        """
        print(f"Arkouda Accessor for DataFrame with {len(self._obj)} rows")
        print("Columns:", list(self._obj.columns))
