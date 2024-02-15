import json
from typing import List, Optional, Union, Literal

import pandas as pd  # type: ignore
from typeguard import typechecked

from arkouda import Categorical, Strings
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import float64 as akfloat64
from arkouda.dtypes import int64 as akint64
from arkouda.groupbyclass import GroupBy, unique
from arkouda.numeric import cast as akcast
from arkouda.pdarrayclass import RegistrationError, pdarray
from arkouda.pdarraycreation import arange, array, create_pdarray, ones
from arkouda.pdarraysetops import argsort, in1d
from arkouda.sorting import coargsort
from arkouda.util import convert_if_categorical, generic_concat, get_callback
from itertools import zip_longest
from collections.abc import  Hashable, Sequence

__all__ = ["Index","MultiIndex"]

class Index:
    objType = "Index"

    @typechecked
    def __init__(
        self,
        values: Union[List, pdarray, Strings, Categorical, pd.Index, "Index"],
        name: Optional[str] = None,
    ):
        self.registered_name: Optional[str] = None
        if isinstance(values, Index):
            self.values = values.values
            self.size = values.size
            self.dtype = values.dtype
            self.name = name if name else values.name
        elif isinstance(values, pd.Index):
            self.values = array(values.values)
            self.size = values.size
            self.dtype = self.values.dtype
            self.name = name if name else values.name
        elif isinstance(values, List):
            values = array(values)
            self.values = values
            self.size = self.values.size
            self.dtype = self.values.dtype
            self.name = name
        elif isinstance(values, (pdarray, Strings, Categorical)):
            self.values = values
            self.size = self.values.size
            self.dtype = self.values.dtype
            self.name = name
        else:
            raise TypeError(f"Unable to create Index from type {type(values)}")

    def __getitem__(self, key):
        from arkouda.series import Series

        if isinstance(key, Series):
            key = key.values

        if isinstance(key, int):
            return self.values[key]

        return Index(self.values[key])

    def __repr__(self):
        # Configured to match pandas
        return f"Index({repr(self.index)}, dtype='{self.dtype}')"

    def __len__(self):
        return len(self.index)

    def __eq__(self, v):
        if isinstance(v, Index):
            return self.index == v.index
        return self.index == v

    @property
    def index(self):
        """
        This is maintained to support older code
        """
        return self.values

    @property
    def shape(self):
        return (self.size,)

    @property
    def is_unique(self):
        """
        Property indicating if all values in the index are unique
        Returns
        -------
            bool - True if all values are unique, False otherwise.
        """
        g = GroupBy(self.values)
        key, ct = g.count()
        return (ct == 1).all()

    @staticmethod
    def factory(index):
        t = type(index)
        if isinstance(index, Index):
            return index
        elif t != list and t != tuple:
            return Index(index)
        else:
            return MultiIndex(index)

    @classmethod
    def from_return_msg(cls, rep_msg):
        data = json.loads(rep_msg)

        idx = []
        for d in data:
            i_comps = d.split("+|+")
            if i_comps[0].lower() == pdarray.objType.lower():
                idx.append(create_pdarray(i_comps[1]))
            elif i_comps[0].lower() == Strings.objType.lower():
                idx.append(Strings.from_return_msg(i_comps[1]))
            elif i_comps[0].lower() == Categorical.objType.lower():
                idx.append(Categorical.from_return_msg(i_comps[1]))

        return cls.factory(idx) if len(idx) > 1 else cls.factory(idx[0])



    def to_pandas(self):
        val = convert_if_categorical(self.values).to_ndarray()
        return pd.Index(data=val, dtype=val.dtype, name=self.name)

    def to_ndarray(self):
        val = convert_if_categorical(self.values)
        return val.to_ndarray()

    def to_list(self):
        return self.to_ndarray().tolist()

    def set_dtype(self, dtype):
        """Change the data type of the index

        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = dtype(self.values)
        self.values = new_idx
        return self

    def register(self, user_defined_name):
        """
        Register this Index object and underlying components with the Arkouda server

        Parameters
        ----------
        user_defined_name : str
            user defined name the Index is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        Index
            The same Index which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Indexes with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Index with the user_defined_name

        See also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")

        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "num_idxs": 1,
                "idx_names": [
                    json.dumps(
                        {
                            "codes": self.values.codes.name,
                            "categories": self.values.categories.name,
                            "NA_codes": self.values._akNAcode.name,
                            **(
                                {"permutation": self.values.permutation.name}
                                if self.values.permutation is not None
                                else {}
                            ),
                            **(
                                {"segments": self.values.segments.name}
                                if self.values.segments is not None
                                else {}
                            ),
                        }
                    )
                    if isinstance(self.values, Categorical)
                    else self.values.name
                ],
                "idx_types": [self.values.objType],
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this Index object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See also
        --------
        register, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    def is_registered(self):
        """
         Return True iff the object is contained in the registry or is a component of a
         registered object.

        Returns
        -------
        numpy.bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mis-match of registered components

        See Also
        --------
        register, attach, unregister

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.util import is_registered

        if self.registered_name is None:
            if not isinstance(self.values, Categorical):
                return is_registered(self.values.name, as_component=True)
            else:
                result = True
                result &= is_registered(self.values.codes.name, as_component=True)
                result &= is_registered(self.values.categories.name, as_component=True)
                result &= is_registered(self.values._akNAcode.name, as_component=True)
                if self.values.permutation is not None and self.values.segments is not None:
                    result &= is_registered(self.values.permutation.name, as_component=True)
                    result &= is_registered(self.values.segments.name, as_component=True)
                return result
        else:
            return is_registered(self.registered_name)

    def to_dict(self, label):
        data = {}
        if label is None:
            label = "idx"
        elif isinstance(label, list):
            label = label[0]
        data[label] = self.index
        return data

    def _check_types(self, other):
        if not isinstance(other, list):
            other = [other]

        for item in other:
            if type(self) is not type(item):
                raise TypeError("Index Types must match")

    def _merge(self, other):
        self._check_types(other)

        callback = get_callback(self.values)
        idx = generic_concat([self.values, other.values], ordered=False)
        return Index(callback(unique(idx)))

    def _merge_all(self, idx_list):
        idx = self.values
        callback = get_callback(idx)

        for other in idx_list:
            self._check_types(other)
            idx = generic_concat([idx, other.values], ordered=False)

        return Index(callback(unique(idx)))

    def _check_aligned(self, other):
        self._check_types(other)
        length = len(self)
        return len(other) == length and (self == other.values).sum() == length

    def argsort(self, ascending=True):
        if not ascending:
            if isinstance(self.values, pdarray) and self.dtype in (akint64, akfloat64):
                i = argsort(-self.values)
            else:
                i = argsort(self.values)[arange(self.size - 1, -1, -1)]
        else:
            i = argsort(self.values)
        return i

    def concat(self, other):
        self._check_types(other)

        if not isinstance(other, list):
            other = [other]

        idx = generic_concat([self.values] + [item.values for item in other], ordered=True)
        return Index(idx)

    def lookup(self, key):
        if not isinstance(key, pdarray):
            # try to handle single value
            try:
                key = array([key])
            except Exception:
                raise TypeError("Lookup must be on an arkouda array")

        return in1d(self.values, key)

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        file_type: str = "distribute",
    ) -> str:
        """
        Save the Index to HDF5.
        The object can be saved to a collection of files or single file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.
        Returns
        -------
        string message indicating result of save operation
        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        Notes
        -----
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`. Otherwise,
        the file name will be `prefix_path`.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        """
        from typing import cast as typecast

        from arkouda.categorical import Categorical as Categorical_
        from arkouda.client import generic_msg
        from arkouda.io import _file_type_to_int, _mode_str_to_int

        index_data = [
            self.values.name
            if not isinstance(self.values, (Categorical_))
            else json.dumps(
                {
                    "codes": self.values.codes.name,
                    "categories": self.values.categories.name,
                    "NA_codes": self.values._akNAcode.name,
                    **(
                        {"permutation": self.values.permutation.name}
                        if self.values.permutation is not None
                        else {}
                    ),
                    **(
                        {"segments": self.values.segments.name}
                        if self.values.segments is not None
                        else {}
                    ),
                }
            )
        ]
        return typecast(
            str,
            generic_msg(
                cmd="tohdf",
                args={
                    "filename": prefix_path,
                    "dset": dataset,
                    "file_format": _file_type_to_int(file_type),
                    "write_mode": _mode_str_to_int(mode),
                    "objType": self.objType,
                    "num_idx": 1,
                    "idx": index_data,
                    "idx_objTypes": [self.values.objType],  # this will be pdarray, strings, or cat
                    "idx_dtypes": [str(self.values.dtype)],
                },
            ),
        )

    def update_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        repack: bool = True,
    ):
        """
        Overwrite the dataset with the name provided with this Index object. If
        the dataset does not exist it is added.

        Parameters
        -----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files
        repack: bool
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        --------
        str - success message if successful

        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the index

        Notes
        ------
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        - Because HDF5 deletes do not release memory, this will create a copy of the
          file with the new data
        """
        from arkouda.categorical import Categorical as Categorical_
        from arkouda.client import generic_msg
        from arkouda.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        index_data = [
            self.values.name
            if not isinstance(self.values, (Categorical_))
            else json.dumps(
                {
                    "codes": self.values.codes.name,
                    "categories": self.values.categories.name,
                    "NA_codes": self.values._akNAcode.name,
                    **(
                        {"permutation": self.values.permutation.name}
                        if self.values.permutation is not None
                        else {}
                    ),
                    **(
                        {"segments": self.values.segments.name}
                        if self.values.segments is not None
                        else {}
                    ),
                }
            )
        ]

        generic_msg(
            cmd="tohdf",
            args={
                "filename": prefix_path,
                "dset": dataset,
                "file_format": _file_type_to_int(file_type),
                "write_mode": _mode_str_to_int("append"),
                "objType": self.objType,
                "num_idx": 1,
                "idx": index_data,
                "idx_objTypes": [self.values.objType],  # this will be pdarray, strings, or cat
                "idx_dtypes": [str(self.values.dtype)],
                "overwrite": True,
            },
        ),

        if repack:
            _repack_hdf(prefix_path)

    def to_parquet(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        compression: Optional[str] = None,
    ):
        """
        Save the Index to Parquet. The result is a collection of files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        compression : str (Optional)
            (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
            Sets the compression type used with Parquet files
        Returns
        -------
        string message indicating result of save operation
        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        Notes
        -----
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`.
        - 'append' write mode is supported, but is not efficient.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        """
        return self.values.to_parquet(prefix_path, dataset=dataset, mode=mode, compression=compression)

    @typechecked
    def to_csv(
        self,
        prefix_path: str,
        dataset: str = "index",
        col_delim: str = ",",
        overwrite: bool = False,
    ):
        """
        Write Index to CSV file(s). File will contain a single column with the pdarray data.
        All CSV Files written by Arkouda include a header denoting data types of the columns.

        Parameters
        -----------
        prefix_path: str
            The filename prefix to be used for saving files. Files will have _LOCALE#### appended
            when they are written to disk.
        dataset: str
            Column name to save the pdarray under. Defaults to "array".
        col_delim: str
            Defaults to ",". Value to be used to separate columns within the file.
            Please be sure that the value used DOES NOT appear in your dataset.
        overwrite: bool
            Defaults to False. If True, any existing files matching your provided prefix_path will
            be overwritten. If False, an error will be returned if existing files are found.

        Returns
        --------
        str reponse message

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server

        Notes
        ------
        - CSV format is not currently supported by load/load_all operations
        - The column delimiter is expected to be the same for column names and data
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline (`\n`) at this time.
        """
        return self.values.to_csv(prefix_path, dataset=dataset, col_delim=col_delim, overwrite=overwrite)

    def save(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        compression: Optional[str] = None,
        file_format: str = "HDF5",
        file_type: str = "distribute",
    ) -> str:
        """
        DEPRECATED
        Save the index to HDF5 or Parquet. The result is a collection of files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        compression : str (Optional)
            (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
            Sets the compression type used with Parquet files
        file_format : str {'HDF5', 'Parquet'}
            By default, saved files will be written to the HDF5 file format. If
            'Parquet', the files will be written to the Parquet file format. This
            is case insensitive.
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.
        Returns
        -------
        string message indicating result of save operation
        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        ValueError
            Raised if there is an error in parsing the prefix path pointing to
            file write location or if the mode parameter is neither truncate
            nor append
        TypeError
            Raised if any one of the prefix_path, dataset, or mode parameters
            is not a string
        See Also
        --------
        save_all, load, read, to_parquet, to_hdf
        Notes
        -----
        The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales``. If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        Previously all files saved in Parquet format were saved with a ``.parquet`` file extension.
        This will require you to use load as if you saved the file with the extension. Try this if
        an older file is not being found.
        Any file extension can be used. The file I/O does not rely on the extension to determine the
        file format.
        """
        from warnings import warn

        warn(
            "ak.Index.save has been deprecated. Please use ak.Index.to_parquet or ak.Index.to_hdf",
            DeprecationWarning,
        )
        if mode.lower() not in ["append", "truncate"]:
            raise ValueError("Allowed modes are 'truncate' and 'append'")

        if file_format.lower() == "hdf5":
            return self.to_hdf(prefix_path, dataset=dataset, mode=mode, file_type=file_type)
        elif file_format.lower() == "parquet":
            return self.to_parquet(prefix_path, dataset=dataset, mode=mode, compression=compression)
        else:
            raise ValueError("Valid file types are HDF5 or Parquet")


    def append(self, other ) :
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        Index

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.append(pd.Index([4]))
        Index([1, 2, 3, 4], dtype='int64')
        """
        to_concat = [self]

        if isinstance(other, (list, tuple)):
            to_concat += list(other)
        else:
            # error: Argument 1 to "append" of "list" has incompatible type
            # "Union[Index, Sequence[Index]]"; expected "Index"
            to_concat.append(other)  # type: ignore[arg-type]

        for obj in to_concat:
            if not isinstance(obj, Index):
                raise TypeError("all inputs must be Index")

        names = {obj.name for obj in to_concat}
        name = None if len(names) > 1 else self.name

        return self._concat(to_concat, name)

    def _concat(self, to_concat, name: Hashable) :
        """
        Concatenate multiple Index objects.
        """
        to_concat_vals = [x.values for x in to_concat]

        from arkouda import concatenate
        result = concatenate(to_concat_vals)

        return result
        #return Index._with_infer(result, name=name)


    # @classmethod
    # def _with_infer(cls, *args, **kwargs):
    #     """
    #     Constructor that uses the 1.0.x behavior inferring numeric dtypes
    #     for ndarray[object] inputs.
    #     """
    #     result = cls(*args, **kwargs)
    #
    #     import numpy as np
    #     if result.dtype == np.dtype("object") and not result._is_multi:
    #         # error: Argument 1 to "maybe_convert_objects" has incompatible type
    #         # "Union[ExtensionArray, ndarray[Any, Any]]"; expected
    #         # "ndarray[Any, Any]"
    #         values = lib.maybe_convert_objects(result._values)  # type: ignore[arg-type]
    #         if values.dtype.kind in "iufb":
    #             return Index(values, name=result.name)
    #
    #     return result

class MultiIndex(Index):
    objType = "MultiIndex"

    def __init__(self, values, name=None, names=None):
        self.registered_name: Optional[str] = None
        if not (isinstance(values, list) or isinstance(values, tuple)):
            raise TypeError("MultiIndex should be an iterable")
        self.values = values
        first = True
        self.names = names
        self.name = name
        for col in self.values:
            # col can be a python int which doesn't have a size attribute
            col_size = col.size if not isinstance(col, int) else 0
            if first:
                # we are implicitly assuming values contains arkouda types and not python lists
                # because we are using obj.size/obj.dtype instead of len(obj)/type(obj)
                # this should be made explict using typechecking
                self.size = col_size
                first = False
            else:
                if col_size != self.size:
                    raise ValueError("All columns in MultiIndex must have same length")
        self.levels = len(self.values)

    def __getitem__(self, key):
        from arkouda.series import Series

        if isinstance(key, Series):
            key = key.values
        return MultiIndex([i[key] for i in self.index])

    def __repr__(self):
        return f"MultiIndex({repr(self.index)})"

    def __len__(self):
        return len(self.index[0])

    def __eq__(self, v):
        if not isinstance(v, (list, tuple, MultiIndex)):
            raise TypeError("Cannot compare MultiIndex to a scalar")
        retval = ones(len(self), dtype=akbool)
        if isinstance(v, MultiIndex):
            v = v.index
        for a, b in zip(self.index, v):
            retval &= a == b

        return retval

    @property
    def index(self):
        return self.values

    def to_pandas(self):
        idx = [convert_if_categorical(i) for i in self.index]
        mi = pd.MultiIndex.from_arrays([i.to_ndarray() for i in idx], names=self.names)
        return pd.Series(index=mi, dtype="float64", name=self.name).index

    def set_dtype(self, dtype):
        """Change the data type of the index

        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = [dtype(i) for i in self.index]
        self.index = new_idx
        return self

    def to_ndarray(self):
        import numpy as np

        return np.array([convert_if_categorical(val).to_ndarray() for val in self.values])

    def to_list(self):
        return self.to_ndarray().tolist()

    def register(self, user_defined_name):
        """
        Register this Index object and underlying components with the Arkouda server

        Parameters
        ----------
        user_defined_name : str
            user defined name the Index is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        MultiIndex
            The same Index which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Indexes with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Index with the user_defined_name

        See also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")

        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "num_idxs": len(self.values),
                "idx_names": [
                    json.dumps(
                        {
                            "codes": v.codes.name,
                            "categories": v.categories.name,
                            "NA_codes": v._akNAcode.name,
                            **({"permutation": v.permutation.name} if v.permutation is not None else {}),
                            **({"segments": v.segments.name} if v.segments is not None else {}),
                        }
                    )
                    if isinstance(v, Categorical)
                    else v.name
                    for v in self.values
                ],
                "idx_types": [v.objType for v in self.values],
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        from arkouda.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    def is_registered(self):
        from arkouda.util import is_registered

        if self.registered_name is None:
            return False
        return is_registered(self.registered_name)

    def to_dict(self, labels):
        data = {}
        if labels is None:
            labels = [f"idx_{i}" for i in range(len(self.index))]
        for i, value in enumerate(self.index):
            data[labels[i]] = value
        return data

    def _merge(self, other):
        self._check_types(other)
        idx = [generic_concat([ix1, ix2], ordered=False) for ix1, ix2 in zip(self.index, other.index)]
        return MultiIndex(GroupBy(idx).unique_keys)

    def _merge_all(self, array):
        idx = self.index

        for other in array:
            self._check_types(other)
            idx = [generic_concat([ix1, ix2], ordered=False) for ix1, ix2 in zip(idx, other.index)]

        return MultiIndex(GroupBy(idx).unique_keys)

    def argsort(self, ascending=True):
        i = coargsort(self.index)
        if not ascending:
            i = i[arange(self.size - 1, -1, -1)]
        return i

    def concat(self, other):
        self._check_types(other)
        idx = [generic_concat([ix1, ix2], ordered=True) for ix1, ix2 in zip(self.index, other.index)]
        return MultiIndex(idx)

    def lookup(self, key):
        if not isinstance(key, list) and not isinstance(key, tuple):
            raise TypeError("MultiIndex lookup failure")
        # if individual vals convert to pdarrays
        if not isinstance(key[0], pdarray):
            dt = self.values[0].dtype if isinstance(self.values[0], pdarray) else akint64
            key = [akcast(array([x]), dt) for x in key]

        return in1d(self.index, key)

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        file_type: str = "distribute",
    ) -> str:
        """
        Save the Index to HDF5.
        The object can be saved to a collection of files or single file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.
        Returns
        -------
        string message indicating result of save operation
        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        Notes
        -----
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`. Otherwise,
        the file name will be `prefix_path`.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        """
        from typing import cast as typecast

        from arkouda.categorical import Categorical as Categorical_
        from arkouda.client import generic_msg
        from arkouda.io import _file_type_to_int, _mode_str_to_int

        index_data = [
            obj.name
            if not isinstance(obj, (Categorical_))
            else json.dumps(
                {
                    "codes": obj.codes.name,
                    "categories": obj.categories.name,
                    "NA_codes": obj._akNAcode.name,
                    **({"permutation": obj.permutation.name} if obj.permutation is not None else {}),
                    **({"segments": obj.segments.name} if obj.segments is not None else {}),
                }
            )
            for obj in self.values
        ]
        return typecast(
            str,
            generic_msg(
                cmd="tohdf",
                args={
                    "filename": prefix_path,
                    "dset": dataset,
                    "file_format": _file_type_to_int(file_type),
                    "write_mode": _mode_str_to_int(mode),
                    "objType": self.objType,
                    "num_idx": len(self.values),
                    "idx": index_data,
                    "idx_objTypes": [obj.objType for obj in self.values],
                    "idx_dtypes": [str(obj.dtype) for obj in self.values],
                },
            ),
        )

    def update_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        repack: bool = True,
    ):
        """
        Overwrite the dataset with the name provided with this Index object. If
        the dataset does not exist it is added.

        Parameters
        -----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files
        repack: bool
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        --------
        str - success message if successful

        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the index

        Notes
        ------
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        - Because HDF5 deletes do not release memory, this will create a copy of the
          file with the new data
        """
        from arkouda.categorical import Categorical as Categorical_
        from arkouda.client import generic_msg
        from arkouda.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        index_data = [
            obj.name
            if not isinstance(obj, (Categorical_))
            else json.dumps(
                {
                    "codes": obj.codes.name,
                    "categories": obj.categories.name,
                    "NA_codes": obj._akNAcode.name,
                    **({"permutation": obj.permutation.name} if obj.permutation is not None else {}),
                    **({"segments": obj.segments.name} if obj.segments is not None else {}),
                }
            )
            for obj in self.values
        ]

        generic_msg(
            cmd="tohdf",
            args={
                "filename": prefix_path,
                "dset": dataset,
                "file_format": _file_type_to_int(file_type),
                "write_mode": _mode_str_to_int("append"),
                "objType": self.objType,
                "num_idx": len(self.values),
                "idx": index_data,
                "idx_objTypes": [obj.objType for obj in self.values],
                "idx_dtypes": [str(obj.dtype) for obj in self.values],
                "overwrite": True,
            },
        ),

        if repack:
            _repack_hdf(prefix_path)


def get_objs_combined_axis(
        objs,
        intersect: bool = False,
        axis: Union[int, Literal["index", "columns", "rows"]] = 0,
        sort: bool = True,
) -> Index:
    """
    Extract combined index: return intersection or union (depending on the
    value of "intersect") of indexes on given axis, or None if all objects
    lack indexes (e.g. they are numpy arrays).

    Parameters
    ----------
    objs : list
        Series or DataFrame objects, may be mix of the two.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    axis : {0 or 'index', 1 or 'outer'}, default 0
        The axis to extract indexes from.
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    obs_idxes = [obj._get_axis(axis) for obj in objs]
    return _get_combined_index(obs_idxes, intersect=intersect, sort=sort)


def _get_distinct_objs(objs: list[Index]) -> list[Index]:
    """
    Return a list with distinct elements of "objs" (different ids).
    Preserves order.
    """
    ids: set[int] = set()
    res = []
    for obj in objs:
        if id(obj) not in ids:
            ids.add(id(obj))
            res.append(obj)
    return res


def _get_combined_index(
        indexes: list[Index],
        intersect: bool = False,
        sort: bool = False,
) -> Index:
    """
    Return the union or intersection of indexes.

    Parameters
    ----------
    indexes : list of Index or list objects
        When intersect=True, do not accept list of lists.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    sort : bool, default False
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    # TODO: handle index names!
    indexes = _get_distinct_objs(indexes)
    if len(indexes) == 0:
        index = Index([])
    elif len(indexes) == 1:
        index = indexes[0]
    elif intersect:
        index = indexes[0]
        for other in indexes[1:]:
            index = index.intersection(other)
    else:
        index = union_indexes(indexes, sort=False)
        index = ensure_index(index)

    if sort:
        index = index.argsort()
    return index



def union_indexes(indexes, sort: bool | None = True) -> Index:
    """
    Return the union of indexes.

    The behavior of sort and names is not consistent.

    Parameters
    ----------
    indexes : list of Index or list objects
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    if len(indexes) == 0:
        raise AssertionError("Must have at least 1 Index to union")
    if len(indexes) == 1:
        result = indexes[0]
        if isinstance(result, list):
            if not sort:
                result = Index(result)
            else:
                result = Index(sorted(result))
        return result

    indexes, kind = _sanitize_and_check(indexes)

    def _unique_indices(inds, dtype) -> Index:
        """
        Concatenate indices and remove duplicates.

        Parameters
        ----------
        inds : list of Index or list objects
        dtype : dtype to set for the resulting Index

        Returns
        -------
        Index
        """
        if all(isinstance(ind, Index) for ind in inds):
            inds = [ind.astype(dtype, copy=False) for ind in inds]
            result = inds[0].unique()
            other = inds[1].append(inds[2:])
            diff = other[result.get_indexer_for(other) == -1]
            if len(diff):
                result = result.append(diff.unique())
            if sort:
                result = result.sort_values()
            return result

        def conv(i):
            if isinstance(i, Index):
                i = i.tolist()
            return i

        return Index(
            lib.fast_unique_multiple_list([conv(i) for i in inds], sort=sort),
            dtype=dtype,
        )

    def _find_common_index_dtype(inds):
        """
        Finds a common type for the indexes to pass through to resulting index.

        Parameters
        ----------
        inds: list of Index or list objects

        Returns
        -------
        The common type or None if no indexes were given
        """
        dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]
        if dtypes:
            dtype = find_common_type(dtypes)
        else:
            dtype = None

        return dtype

    if kind == "special":
        result = indexes[0]

        dtis = [x for x in indexes if isinstance(x, DatetimeIndex)]
        dti_tzs = [x for x in dtis if x.tz is not None]
        if len(dti_tzs) not in [0, len(dtis)]:
            # TODO: this behavior is not tested (so may not be desired),
            #  but is kept in order to keep behavior the same when
            #  deprecating union_many
            # test_frame_from_dict_with_mixed_indexes
            raise TypeError("Cannot join tz-naive with tz-aware DatetimeIndex")

        if len(dtis) == len(indexes):
            sort = True
            result = indexes[0]

        elif len(dtis) > 1:
            # If we have mixed timezones, our casting behavior may depend on
            #  the order of indexes, which we don't want.
            sort = False

            # TODO: what about Categorical[dt64]?
            # test_frame_from_dict_with_mixed_indexes
            indexes = [x.astype(object, copy=False) for x in indexes]
            result = indexes[0]

        for other in indexes[1:]:
            result = result.union(other, sort=None if sort else False)
        return result

    elif kind == "array":
        dtype = _find_common_index_dtype(indexes)
        index = indexes[0]
        if not all(index.equals(other) for other in indexes[1:]):
            index = _unique_indices(indexes, dtype)

        name = get_unanimous_names(*indexes)[0]
        if name != index.name:
            index = index.rename(name)
        return index
    else:  # kind='list'
        dtype = _find_common_index_dtype(indexes)
        return _unique_indices(indexes, dtype)


def _sanitize_and_check(indexes):
    """
    Verify the type of indexes and convert lists to Index.

    Cases:

    - [list, list, ...]: Return ([list, list, ...], 'list')
    - [list, Index, ...]: Return _sanitize_and_check([Index, Index, ...])
        Lists are sorted and converted to Index.
    - [Index, Index, ...]: Return ([Index, Index, ...], TYPE)
        TYPE = 'special' if at least one special type, 'array' otherwise.

    Parameters
    ----------
    indexes : list of Index or list objects

    Returns
    -------
    sanitized_indexes : list of Index or list objects
    type : {'list', 'array', 'special'}
    """
    kinds = list({type(index) for index in indexes})

    if list in kinds:
        if len(kinds) > 1:
            indexes = [
                Index(list(x)) if not isinstance(x, Index) else x for x in indexes
            ]
            kinds.remove(list)
        else:
            return indexes, "list"

    if len(kinds) > 1 or Index not in kinds:
        return indexes, "special"
    else:
        return indexes, "array"


def all_indexes_same(indexes) -> bool:
    """
    Determine if all indexes contain the same elements.

    Parameters
    ----------
    indexes : iterable of Index objects

    Returns
    -------
    bool
        True if all indexes contain the same elements, False otherwise.
    """
    itr = iter(indexes)
    first = next(itr)
    return all(first.equals(index) for index in itr)


def default_index(n: int) -> Index:
    from arkouda import arange
    return Index(arange(n))

def ensure_index(index_like: Union[int, Literal["index", "columns", "rows"]], copy: bool = False) -> Index:
    """
    Ensure that we have an index from some index-like object.

    Parameters
    ----------
    index_like : sequence
        An Index or other sequence
    copy : bool, default False

    Returns
    -------
    index : Index or MultiIndex

    See Also
    --------
    ensure_index_from_sequences

    Examples
    --------
    >>> ensure_index(["a", "b"])
    Index(['a', 'b'], dtype='object')

    >>> ensure_index([("a", "a"), ("b", "c")])
    Index([('a', 'a'), ('b', 'c')], dtype='object')

    >>> ensure_index([["a", "a"], ["b", "c"]])
    MultiIndex([('a', 'b'),
            ('a', 'c')],
           )
    """
    if isinstance(index_like, Index):
        if copy:
            index_like = index_like.copy()
        return index_like

    if isinstance(index_like, ABCSeries):
        name = index_like.name
        return Index(index_like, name=name, copy=copy)

    if is_iterator(index_like):
        index_like = list(index_like)

    if isinstance(index_like, list):
        if type(index_like) is not list:  # noqa: E721
            # must check for exactly list here because of strict type
            # check in clean_index_list
            index_like = list(index_like)

        if len(index_like) and lib.is_all_arraylike(index_like):
            from pandas.core.indexes.multi import MultiIndex

            return MultiIndex.from_arrays(index_like)
        else:
            return Index(index_like, copy=copy, tupleize_cols=False)
    else:
        return Index(index_like, copy=copy)

def get_unanimous_names(*indexes: Index) -> tuple[Hashable, ...]:
    """
    Return common name if all indices agree, otherwise None (level-by-level).

    Parameters
    ----------
    indexes : list of Index objects

    Returns
    -------
    list
        A list representing the unanimous 'names' found.
    """
    name_tups = [tuple(i.names) for i in indexes]
    name_sets = [{*ns} for ns in zip_longest(*name_tups)]
    names = tuple(ns.pop() if len(ns) == 1 else None for ns in name_sets)
    return names