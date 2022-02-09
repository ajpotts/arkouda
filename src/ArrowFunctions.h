#include <stdint.h>

// Wrap functions in C extern if compiling C++ object file
#ifdef __cplusplus
#include <iostream>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/column_reader.h>
#include <parquet/api/writer.h>
extern "C" {
#endif

#define ARROWINT64 0
#define ARROWINT32 1
#define ARROWUINT64 2
#define ARROWTIMESTAMP ARROWINT64
#define ARROWUNDEFINED -1
#define ARROWERROR -1

  // Each C++ function contains the actual implementation of the
  // functionality, and there is a corresponding C function that
  // Chapel can call into through C interoperability, since there
  // is no C++ interoperability supported in Chapel today.
  int64_t c_getNumRows(const char*, char** errMsg);
  int64_t cpp_getNumRows(const char*, char** errMsg);

  int c_readColumnByName(const char* filename, void* chpl_arr,
                         const char* colname, int64_t numElems, int64_t batchSize,
                         char** errMsg);
  int cpp_readColumnByName(const char* filename, void* chpl_arr,
                           const char* colname, int64_t numElems, int64_t batchSize,
                           char** errMsg);

  int c_getType(const char* filename, const char* colname, char** errMsg);
  int cpp_getType(const char* filename, const char* colname, char** errMsg);

  int cpp_writeColumnToParquet(const char* filename, void* chpl_arr,
                               int64_t colnum, const char* dsetname, int64_t numelems,
                               int64_t rowGroupSize, int64_t dtype, char** errMsg);
  int c_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int64_t colnum, const char* dsetname, int64_t numelems,
                             int64_t rowGroupSize, int64_t dtype, char** errMsg);
    
  const char* c_getVersionInfo(void);
  const char* cpp_getVersionInfo(void);

  void c_free_string(void* ptr);
  void cpp_free_string(void* ptr);
  
#ifdef __cplusplus
}
#endif