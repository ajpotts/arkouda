import os, shutil, glob
import numpy as np
from typing import List, Mapping, Union
from base_test import ArkoudaTest
from context import arkouda as ak
from arkouda import io_util

'''
Tests writting Arkouda pdarrays to and from files
'''
class IOTest(ArkoudaTest):

    @classmethod
    def setUpClass(cls):
        super(IOTest, cls).setUpClass()
        IOTest.io_test_dir = '{}/io_test'.format(os.getcwd())
        io_util.get_directory(IOTest.io_test_dir)

    def setUp(self):
        ArkoudaTest.setUp(self)
        self.int_tens_pdarray = ak.array(np.random.randint(-100,100,1000))
        self.int_tens_pdarray_dupe = ak.array(np.random.randint(-100,100,1000))
        self.int_hundreds_pdarray = ak.array(np.random.randint(-1000,1000,1000))
        self.int_hundreds_pdarray_dupe = ak.array(np.random.randint(-1000,1000,1000))
        self.float_pdarray = ak.array(np.random.uniform(-100,100,1000))  
        self.float_pdarray_dupe = ak.array(np.random.uniform(-100,100,1000))   
        
        self.dict_columns =  {
           'int_tens_pdarray' : self.int_tens_pdarray,
           'int_hundreds_pdarray' : self.int_hundreds_pdarray,
           'float_pdarray' : self.float_pdarray
        }
        
        self.dict_columns_dupe =  {
           'int_tens_pdarray' : self.int_tens_pdarray_dupe,
           'int_hundreds_pdarray' : self.int_hundreds_pdarray_dupe,
           'float_pdarray' : self.float_pdarray_dupe
        }
        
        self.dict_single_column = {
           'int_tens_pdarray' : self.int_tens_pdarray
        }
        
        self.list_columns = [
          self.int_tens_pdarray,
          self.int_hundreds_pdarray,
          self.float_pdarray
        ]
        
        self.names =  [
          'int_tens_pdarray',
          'int_hundreds_pdarray',
          'float_pdarray'
        ]

    def _create_file(self, prefix_path : str, columns : Union[Mapping[str,ak.array]], 
                                           names : List[str]=None) -> None:
        '''
        Creates an hdf5 file with dataset(s) from the specified columns and path prefix
        via the ak.save_all method. If columns is a List, then the names list is used 
        to create the datasets
        
        :return: None
        :raise: ValueError if the names list is None when columns is a list
        '''       
        if isinstance(columns, dict):
            ak.save_all(columns=columns, prefix_path=prefix_path)   
        else:
            if not names:
                raise ValueError('the names list must be not None if columns is a list')
            ak.save_all(columns=columns, prefix_path=prefix_path, names=names)
    
    def testSaveAllLoadAllWithDict(self): 

        '''
        Creates 2..n files from an input columns dict depending upon the number of 
        arkouda_server locales, retrieves all datasets and correspoding pdarrays, 
        and confirms they match inputs
        
        :return: None
        :raise: AssertionError if the input and returned datasets and pdarrays don't match
        '''
        self._create_file(columns=self.dict_columns, prefix_path='{}/iotest_dict'.format(IOTest.io_test_dir))
        retrieved_columns = ak.load_all('{}/iotest_dict'.format(IOTest.io_test_dir))

        self.assertEqual(3, len(retrieved_columns))
        self.assertEqual(self.dict_columns['int_tens_pdarray'].all(), 
                         retrieved_columns['int_tens_pdarray'].all())
        self.assertEqual(self.dict_columns['int_hundreds_pdarray'].all(), 
                         retrieved_columns['int_hundreds_pdarray'].all())
        self.assertEqual(self.dict_columns['float_pdarray'].all(), 
                         retrieved_columns['float_pdarray'].all())      
        self.assertEqual(3, len(ak.get_datasets('{}/iotest_dict_LOCALE0'.format(IOTest.io_test_dir))))
        
    def testSaveAllLoadAllWithList(self):
        '''
        Creates 2..n files from an input columns and names list depending upon the number of 
        arkouda_server locales, retrieves all datasets and correspoding pdarrays, and confirms 
        they match inputs
        
        :return: None
        :raise: AssertionError if the input and returned datasets and pdarrays don't match
        '''
        self._create_file(columns=self.list_columns, prefix_path='{}/iotest_list'.format(IOTest.io_test_dir), 
                          names=self.names)
        retrieved_columns = ak.load_all(path_prefix='{}/iotest_list'.format(IOTest.io_test_dir))

        self.assertEqual(3, len(retrieved_columns))
        self.assertEqual(self.list_columns[0].all(), 
                         retrieved_columns['int_tens_pdarray'].all())
        self.assertEqual(self.list_columns[1].all(), 
                         retrieved_columns['int_hundreds_pdarray'].all())
        self.assertEqual(self.list_columns[2].all(), 
                         retrieved_columns['float_pdarray'].all())      
        self.assertEqual(3, len(ak.get_datasets('{}/iotest_list_LOCALE0'.format(IOTest.io_test_dir))))
    
    def testLsHdf(self):
        '''
        Creates 1..n files depending upon the number of arkouda_server locales, invokes the 
        ls_hdf method on an explicit file name reads the files and confirms the expected 
        message was returned.

        :return: None
        :raise: AssertionError if the h5ls output does not match expected value
        '''
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column'.format(IOTest.io_test_dir))
        message = ak.ls_hdf('{}/iotest_single_column_LOCALE0'.format(IOTest.io_test_dir))
        self.assertIn('int_tens_pdarray         Dataset', message)

    def testReadHdf(self):
        '''
        Creates 2..n files depending upon the number of arkouda_server locales with two
        files each containing different-named datasets with the same pdarrays, reads the files
        with an explicit list of file names to the read_hdf method, and confirms the dataset 
        was returned correctly.

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        '''
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column'.format(IOTest.io_test_dir))
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column_dupe'.format(IOTest.io_test_dir))
        
        dataset = ak.read_hdf(dsetName='int_tens_pdarray', 
                    filenames=['{}/iotest_single_column_LOCALE0'.format(IOTest.io_test_dir),
                               '{}/iotest_single_column_dupe_LOCALE0'.format(IOTest.io_test_dir)])
        self.assertIsNotNone(dataset)
        
    def testReadHdfWithGlob(self):
        '''
        Creates 2..n files depending upon the number of arkouda_server locales with two
        files each containing different-named datasets with the same pdarrays, reads the files
        with the glob feature of the read_hdf method, and confirms the datasets and embedded 
        pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        '''
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column'.format(IOTest.io_test_dir))
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column_dupe'.format(IOTest.io_test_dir))
        
        dataset = ak.read_hdf(dsetName='int_tens_pdarray', 
                    filenames='{}/iotest_single_column*'.format(IOTest.io_test_dir))
        self.assertEqual(self.int_tens_pdarray.all(), dataset.all())

    def testReadAll(self):
        '''
        Creates 2..n files depending upon the number of arkouda_server locales, reads the files
        with an explicit list of file names to the read_all method, and confirms the datasets 
        and embedded pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns'.format(IOTest.io_test_dir))
        
        dataset = ak.read_all(filenames=['{}/iotest_dict_columns_LOCALE0'.format(IOTest.io_test_dir)])
        self.assertEqual(3, len(list(dataset.keys())))     
        
    def testReadAllWithGlob(self):
        '''
        Creates 2..n files depending upon the number of arkouda_server locales with two
        files each containing different-named datasets with the same pdarrays, reads the files
        with the glob feature of the read_all method, and confirms the datasets and embedded 
        pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns'.format(IOTest.io_test_dir))
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns_dupe'.format(IOTest.io_test_dir))
        
        dataset = ak.read_all(filenames='{}/iotest_dict_columns*'.format(IOTest.io_test_dir))

        self.assertEqual(3, len(list(dataset.keys())))  
        self.assertEqual(self.int_tens_pdarray.all(), dataset['int_tens_pdarray'].all())
        self.assertEqual(self.int_hundreds_pdarray.all(), dataset['int_hundreds_pdarray'].all())
        self.assertEqual(self.float_pdarray.all(), dataset['float_pdarray'].all())

    def testLoad(self):
        '''
        Creates 1..n files depending upon the number of arkouda_server locales with three columns 
        AKA datasets, loads each corresponding dataset and confirms each corresponding pdarray 
        equals the input pdarray.
        
        :return: None
        :raise: AssertionError if the input and returned datasets (pdarrays) don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns'.format(IOTest.io_test_dir)) 
        result_array_tens = ak.load(path_prefix='{}/iotest_dict_columns'.format(IOTest.io_test_dir), 
                                    dataset='int_tens_pdarray')
        result_array_hundreds = ak.load(path_prefix='{}/iotest_dict_columns'.format(IOTest.io_test_dir), 
                                        dataset='int_hundreds_pdarray')
        result_array_float = ak.load(path_prefix='{}/iotest_dict_columns'.format(IOTest.io_test_dir), 
                                     dataset='float_pdarray')

        self.assertEqual(self.int_tens_pdarray.all(), result_array_tens.all())
        self.assertEqual(self.int_hundreds_pdarray.all(), result_array_hundreds.all())
        self.assertEqual(self.float_pdarray.all(), result_array_float.all())
        
    def testGetDataSets(self):
        '''
        Creates 1..n files depending upon the number of arkouda_server locales containing three 
        datasets and confirms the expected number of datasets along with the dataset names
        
        :return: None
        :raise: AssertionError if the input and returned dataset names don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns'.format(IOTest.io_test_dir))     
        datasets = ak.get_datasets('{}/iotest_dict_columns_LOCALE0'.format(IOTest.io_test_dir))

        self.assertEqual(3, len(datasets)) 
        for dataset in datasets:
            self.assertIn(dataset, self.names)

    def testSaveStringsDataset(self):
        # Create, save, and load Strings dataset
        strings_array = ak.array(['testing string{}'.format(num) for num in list(range(1,11))])
        strings_array.save('{}/strings-test'.format(IOTest.io_test_dir), dataset='strings')
        r_strings_array = ak.load('{}/strings-test'.format(IOTest.io_test_dir), 
                                  dataset='strings')
        self.assertTrue((strings_array == r_strings_array).all())

        # Read a part of a saved Strings dataset from one hdf5 file
        r_strings_subset = ak.read_all(filenames='{}/strings-test_LOCALE0'.\
                                    format(IOTest.io_test_dir))
        self.assertIsNotNone(r_strings_subset)
        self.assertTrue(isinstance(r_strings_subset[0], str))    
        self.assertIsNotNone(ak.read_hdf(filenames='{}/strings-test_LOCALE0'.\
                            format(IOTest.io_test_dir), dsetName='strings/values'))
        self.assertIsNotNone(ak.read_hdf(filenames='{}/strings-test_LOCALE0'.\
                            format(IOTest.io_test_dir), dsetName='strings/segments'))
        
    def testSaveLongStringsDataset(self):
        # Create, save, and load Strings dataset
        strings_array = ak.array(['testing a longer string{} to be written, loaded and appended'.\
                                  format(num) for num in list(range(1,11))])
        strings_array.save('{}/strings-test'.format(IOTest.io_test_dir), dataset='strings')
        r_strings_array = ak.load('{}/strings-test'.format(IOTest.io_test_dir), 
                                  dataset='strings')
        self.assertTrue((strings_array == r_strings_array).all())       

    def testSaveMixedStringsDataset(self):
        strings_array = ak.array(['string {}'.format(num) for num in list(range(1,11))])
        m_floats =  ak.array([x / 10.0 for x in range(0, 10)])      
        m_ints = ak.array(list(range(0, 10)))
        ak.save_all({'m_strings': strings_array,
                     'm_floats' : m_floats,
                     'm_ints' : m_ints}, 
                     '{}/multi-type-test'.format(IOTest.io_test_dir))
        r_mixed = ak.load_all('{}/multi-type-test'.format(IOTest.io_test_dir))
        self.assertTrue((strings_array == r_mixed['m_strings']).all())
        self.assertTrue((m_floats == r_mixed['m_floats']).all())
        self.assertTrue((m_ints == r_mixed['m_ints']).all())

        r_floats = ak.load('{}/multi-type-test'.format(IOTest.io_test_dir), 
                           dataset='m_floats')
        self.assertTrue((m_floats == r_floats).all())
        r_strings = ak.load('{}/multi-type-test'.format(IOTest.io_test_dir), 
                            dataset='m_strings')
        self.assertTrue((strings_array == r_strings).all())

    def testAppendStringsDataset(self):
        strings_array = ak.array(['string {}'.format(num) for num in list(range(1,11))])
        strings_array.save('{}/append-strings-test'.format(IOTest.io_test_dir), 
                           dataset='strings')
        strings_array.save('{}/append-strings-test'.format(IOTest.io_test_dir), 
                           dataset='strings-dupe', mode='append')

        r_strings = ak.load('{}/append-strings-test'.format(IOTest.io_test_dir), 
                                 dataset='strings')
        r_strings_dupe = ak.load('{}/append-strings-test'.format(IOTest.io_test_dir), 
                                 dataset='strings-dupe')  
        self.assertTrue((r_strings == r_strings_dupe).all())

    def testAppendMixedStringsDataset(self):
        strings_array = ak.array(['string {}'.format(num) for num in list(range(1,11))])
        strings_array.save('{}/append-multi-type-test'.format(IOTest.io_test_dir), 
                           dataset='m_strings') 
        m_floats =  ak.array([x / 10.0 for x in range(0, 10)])      
        m_ints = ak.array(list(range(0, 10)))
        ak.save_all({'m_floats' : m_floats,
                     'm_ints' : m_ints}, 
                     '{}/append-multi-type-test'.format(IOTest.io_test_dir), mode='append')
        r_mixed = ak.load_all('{}/append-multi-type-test'.format(IOTest.io_test_dir))
      
        self.assertTrue((strings_array == r_mixed['m_strings']).all())   
        self.assertTrue((m_floats == r_mixed['m_floats']).all())
        self.assertTrue((m_ints == r_mixed['m_ints']).all())

    def tearDown(self):
        super(IOTest, self).tearDown()
        for f in glob.glob('{}/*'.format(IOTest.io_test_dir)):
            os.remove(f)

    @classmethod
    def tearDownClass(cls):
        super(IOTest, cls).tearDownClass()
        shutil.rmtree(IOTest.io_test_dir)
