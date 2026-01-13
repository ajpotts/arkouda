#include "ReadParquet.h"

#include <arrow/compute/api.h>
#include <arrow/dataset/api.h>
#include <arrow/dataset/file_base.h>
#include <arrow/filesystem/localfs.h>
#include <cstring>

namespace {
  namespace ds = arrow::dataset;
  namespace fs = arrow::fs;

  inline std::string to_lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(::tolower(c));
    return s;
  }
  inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
  }

  arrow::Result<std::shared_ptr<ds::Dataset>> MakeDatasetFromPath(const std::string& path) {
    auto filesystem = std::make_shared<fs::LocalFileSystem>();
    std::shared_ptr<ds::FileFormat> format;
    const std::string lp = to_lower(path);
    if (ends_with(lp, ".parquet") || ends_with(lp, ".pq")) {
      format = std::make_shared<ds::ParquetFileFormat>();
    } else if (ends_with(lp, ".csv") || ends_with(lp, ".tsv")) {
      auto csv_format = std::make_shared<ds::CsvFileFormat>();
      if (ends_with(lp, ".tsv")) {
        csv_format->parse_options.delimiter = '\t';
      }
      format = csv_format;
    } else {
      return arrow::Status::Invalid("Unsupported file extension (expected .parquet/.pq or .csv/.tsv): ", path);
    }

    ds::FileSystemFactoryOptions options;
    ARROW_ASSIGN_OR_RAISE(auto factory,
                          ds::FileSystemDatasetFactory::Make(filesystem,
                                                            std::vector<std::string>{path},
                                                            format,
                                                            options));
    return factory->Finish();
  }

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> MakeSingleColumnReader(
      const std::shared_ptr<ds::Dataset>& dataset,
      const std::string& colname,
      int64_t batch_size) {
    ARROW_ASSIGN_OR_RAISE(auto scan_builder, dataset->NewScan());
    ARROW_RETURN_NOT_OK(scan_builder->Project({colname}));
    ARROW_RETURN_NOT_OK(scan_builder->BatchSize(batch_size));
    ARROW_ASSIGN_OR_RAISE(auto scanner, scan_builder->Finish());
    return scanner->ToRecordBatchReader();
  }

  template <typename T>
  inline void write_value(void* out, int64_t out_i, T v) {
    static_cast<T*>(out)[out_i] = v;
  }

  // Specializations for writing into Chapel's widened buffers.
  template <>
  inline void write_value<int32_t>(void* out, int64_t out_i, int32_t v) {
    static_cast<int64_t*>(out)[out_i] = static_cast<int64_t>(v);
  }
  template <>
  inline void write_value<uint32_t>(void* out, int64_t out_i, uint32_t v) {
    static_cast<uint64_t*>(out)[out_i] = static_cast<uint64_t>(v);
  }
  template <>
  inline void write_value<float>(void* out, int64_t out_i, float v) {
    static_cast<double*>(out)[out_i] = static_cast<double>(v);
  }

}


namespace akcpp {

namespace {

// ChplBuf::ElemType can be used to get the type of the Chapel buffer from an
// Arrow type.
template<typename ArrowType>
struct ChplBuf {
  using ElemType = typename ArrowType::c_type;
};

// specialization for storing 32-bit data in 64-bit arrays
template<> struct ChplBuf<arrow::Int32Type>  { using ElemType = int64_t; };
template<> struct ChplBuf<arrow::UInt32Type> { using ElemType = uint64_t; };
template<> struct ChplBuf<arrow::FloatType>  { using ElemType = double; };

// TODO The implementation seemingly reads 128-bit decimal values into
// Chapel `real`s (or C `double`s). I don't see how this can work properly with
// large enough data. For now, I think this keeps the implementation practically
// same as the old, likely carrying the same bug into the new implementation,
// unfortunately.
// See https://github.com/Bears-R-Us/arkouda/issues/4911.
template<> struct ChplBuf<parquet::FLBAType>  {using ElemType = double;};

// this struct is used to bundle different types into one, based on a single
// ArrowType
template<typename ArrowType>
struct TypeBundle {
  using ReaderType = typename parquet::TypedColumnReader<ArrowType>;
  using ChplType = typename ChplBuf<ArrowType>::ElemType;
  using PqType = typename ArrowType::c_type;
};

template<>
struct TypeBundle<arrow::Decimal128Type> {
  using ReaderType = typename parquet::TypedColumnReader<parquet::FLBAType>;
  using ChplType = typename ChplBuf<parquet::FLBAType>::ElemType;
  using PqType = typename parquet::FixedLenByteArray;
};
} // end anonymous namespace

// Generic read implementation
template<typename ArrowType>
int64_t ColReadOp::read() {
  using Types = TypeBundle<ArrowType>;
  using ChplType = typename Types::ChplType;
  using ReaderType = typename Types::ReaderType;

  auto chpl_ptr = (ChplType*)chpl_arr;
  int64_t num_read = 0;
  ReaderType* reader = static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  // TODO find better variable names. values_read and num_read are confusing
  int64_t values_read = 0;

  // note that floats are the types that natively support nulls. This generic
  // read() shouldn't do anything with floats
  if (nullMode == noNulls || nullMode == onlyFloats) {
    while (reader->HasNext() && row_idx < numElems) {
      if((numElems - row_idx) < batchSize) // adjust batchSize if needed
        batchSize = numElems - row_idx;
      std::ignore = reader->ReadBatch(batchSize, nullptr, nullptr,
                                      &chpl_ptr[row_idx], &values_read);
      row_idx+=values_read;
      num_read+=values_read;
    }
  }
  else {
    int16_t definition_level; // nullable type and only reading single records in batch
    while (reader->HasNext() && row_idx < numElems) {
      std::ignore = reader->ReadBatch(1, &definition_level, nullptr,
                                      &chpl_ptr[row_idx], &values_read);
      // if values_read is 0, that means that it was a null value
      if(values_read == 0) {
        where_null_chpl[row_idx] = true;
      }
      row_idx++;
      num_read++;
    }
  }
  return num_read;
}

template<typename Types>
int64_t ColReadOp::_readShortIntegral() {
  using ReaderType = typename Types::ReaderType;
  using ChplType = typename Types::ChplType;
  using PqType = typename Types::PqType;

  auto chpl_ptr = (ChplType*)chpl_arr;
  ReaderType* reader = static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  // TODO find better variable names. values_read and num_read are confusing
  int64_t values_read = 0;

  int64_t num_read = 0;
  if (not hasNonFloatNulls) {
    // TODO we can read into the actual Chapel array, and then space the data
    // out in reverse order. I don't think we need to have this temporary
    // buffer
    PqType* tmpArr = (PqType*)malloc(batchSize * sizeof(PqType));
    while (reader->HasNext() && row_idx < numElems) {
      if((numElems - row_idx) < batchSize) // adjust batchSize if needed
        batchSize = numElems - row_idx;

      // Can't read directly into chpl_ptr because it is int64
      std::ignore = reader->ReadBatch(batchSize, nullptr, nullptr, tmpArr,
                                      &values_read);

      for (int64_t j = 0; j < values_read; j++) {
        chpl_ptr[row_idx+j] = (ChplType)tmpArr[j];
      }

      row_idx+=values_read;
      num_read+=values_read;
    }
    free(tmpArr);
  }
  else {
    PqType tmp;
    // Engin: we don't seem to use this anywhere, but passing nullptr instead
    // of this causes errors. Why?
    int16_t definition_level; // nullable type and only reading single
                              // records in batch
    while (reader->HasNext() && row_idx < numElems) {
      std::ignore = reader->ReadBatch(1, &definition_level, nullptr, &tmp,
                                      &values_read);
      // if values_read is 0, that means that it was a null value
      if(values_read == 0) {
        where_null_chpl[row_idx] = true;
      }
      else {
        chpl_ptr[row_idx] = (ChplType)tmp;
      }
      row_idx++;
      num_read++;
    }
  }
  return num_read;
}

template<>
int64_t ColReadOp::read<arrow::Int32Type>() {
  return _readShortIntegral<TypeBundle<arrow::Int32Type>>();
}

template<>
int64_t ColReadOp::read<arrow::UInt32Type>() {
  return _readShortIntegral<TypeBundle<arrow::UInt32Type>>();
}

template<>
int64_t ColReadOp::read<arrow::FloatType>() {
  using Types = TypeBundle<arrow::FloatType>;
  using ReaderType = typename Types::ReaderType;
  using ChplType = typename Types::ChplType;
  using PqType = typename Types::PqType;

  auto chpl_ptr = (ChplType*)chpl_arr;
  ReaderType* reader = static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  // TODO find better variable names. values_read and num_read are confusing
  int64_t values_read = 0;

  int64_t num_read = 0;
  if (nullMode == noNulls) {
    PqType* tmpArr = (PqType*)malloc(batchSize * sizeof(PqType));
    while (reader->HasNext() && row_idx < numElems) {
      if((numElems - row_idx) < batchSize) // adjust batchSize if needed
        batchSize = numElems - row_idx;
      std::ignore = reader->ReadBatch(batchSize, nullptr, nullptr, tmpArr,
                                      &values_read);

      // promote to larger type
      for (int64_t j = 0; j < values_read; j++) {
        chpl_ptr[row_idx+j] = (ChplType)tmpArr[j];
      }

      row_idx+=values_read;
      num_read+=values_read;

    }
    free(tmpArr);
  }
  else {
    int16_t definition_level; // nullable type and only reading single records in batch
    while (reader->HasNext() && row_idx < numElems) {
      PqType value = 0;
      std::ignore = reader->ReadBatch(1, &definition_level, nullptr, &value,
                                      &values_read);
      // if values_read is 0, that means that it was a null value
      if(values_read > 0) {
        chpl_ptr[row_idx] = (ChplType) value;
      }
      else {
        chpl_ptr[row_idx] = NAN;
      }
      row_idx++;
      num_read++;
    }
  }
  return num_read;
}

template<>
int64_t ColReadOp::read<arrow::DoubleType>() {
  using Types = TypeBundle<arrow::DoubleType>;
  using ReaderType = typename Types::ReaderType;
  using ChplType = typename Types::ChplType;
  using PqType = typename Types::PqType;

  auto chpl_ptr = (ChplType*)chpl_arr;
  ReaderType* reader = static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  // TODO find better variable names. values_read and num_read are confusing
  int64_t values_read = 0;

  int64_t num_read = 0;
  if (nullMode == noNulls) {
    while (reader->HasNext() && row_idx < numElems) {
      if((numElems - row_idx) < batchSize) // adjust batchSize if needed
        batchSize = numElems - row_idx;
      std::ignore = reader->ReadBatch(batchSize, nullptr, nullptr,
                                      &chpl_ptr[row_idx], &values_read);
      row_idx+=values_read;
      num_read+=values_read;
    }
  }
  else {
    int16_t definition_level; // nullable type and only reading single records in batch
    while (reader->HasNext() && row_idx < numElems) {
      PqType value = 0;
      std::ignore = reader->ReadBatch(1, &definition_level, nullptr, &value,
                                      &values_read);
      // if values_read is 0, that means that it was a null value
      if(values_read > 0) {
        chpl_ptr[row_idx] = (ChplType) value;
      }
      else {
        chpl_ptr[row_idx] = NAN;
      }
      row_idx++;
      num_read++;
    }
  }
  return num_read;
}

// For Parquet's "Decimal" types -- typically larger, or more precise floats
template<>
int64_t ColReadOp::read<arrow::Decimal128Type>() {
  using Types = TypeBundle<arrow::Decimal128Type>;
  using ReaderType = typename Types::ReaderType;
  using ChplType = typename Types::ChplType;
  using PqType = typename Types::PqType;

  auto chpl_ptr = (ChplType*)chpl_arr;
  ReaderType* reader = static_cast<ReaderType*>(column_reader.get());
  startIdx -= reader->Skip(*startIdx);

  // TODO find better variable names. values_read and num_read are confusing
  int64_t values_read = 0;
  int64_t num_read = 0;

  using LogType = parquet::DecimalLogicalType;
  const auto& type = dynamic_cast<const LogType&>(*col_info->logical_type());
  const int64_t precision = type.precision();

  // In ReadParquet.cpp, there is a basic look up table for this. But number
  // of required bytes can be found mathematically:
  int numbits = ceil(precision*3.321928);  // the magic number is the constant
                                           // from log2(10^precision)
  if (numbits%8==0) numbits++;             // add a bit for the sign bit if we
                                           // are at the byte boundary
  const auto numbytes = ceil(numbits/8.0);

  while (reader->HasNext() && row_idx < numElems) {
    PqType value;
    std::ignore = reader->ReadBatch(1, nullptr, nullptr, &value,
                                    &values_read);
    arrow::Decimal128 v;
    PARQUET_ASSIGN_OR_THROW(v,
                            ::arrow::Decimal128::FromBigEndian(value.ptr,
                                                               numbytes));

    chpl_ptr[row_idx] = v.ToDouble(0);
    row_idx+=values_read;
    num_read+=values_read;
  }

  return num_read;
}


int readAllCols(const char* filename, void** chpl_arrs, int* types,
                bool* where_null_chpl, int64_t numElems, int64_t startIdx,
                int64_t batchSize,
                chplEnum_t nullMode, char** errMsg) {
  try {
    std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
        parquet::ParquetFileReader::OpenFile(filename, false);

    std::shared_ptr<parquet::FileMetaData> file_metadata =
        parquet_reader->metadata();

    int num_row_groups = file_metadata->num_row_groups();

    const auto num_cols = cpp_getNumCols(filename, errMsg);
    if (num_cols == ARROWERROR) {
      return ARROWERROR;
    }

    std::vector<int64_t> startIdxPerCol;
    startIdxPerCol.resize(num_cols, startIdx);

    int64_t row_idx = 0;

    for (int rg_idx = 0; (rg_idx<num_row_groups) && (row_idx<numElems);
         rg_idx++) {

      std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(rg_idx);

      std::shared_ptr<parquet::ColumnReader> column_reader;

      int64_t nrows_in_iter = -1;

      for (int col_idx = 0; col_idx<num_cols ; col_idx++) {
        column_reader = row_group_reader->Column(col_idx);
        const parquet::ColumnDescriptor* col_info =
          file_metadata->schema()->Column(col_idx);

        void* chpl_arr = chpl_arrs[col_idx];

        // TODO, I want values of this type to be created per read operation,
        // not per column, per rowgroup
        auto op = ColReadOp { chpl_arr,
                              &startIdxPerCol[col_idx],
                              column_reader,
                              false, // has_non_float_nulls is for backward
                                     // compat and ignored while reading all
                                     // columns. nullMode is used instead.
                              nullMode,
                              row_idx,
                              numElems,
                              batchSize,
                              where_null_chpl,
                              col_info };

        int64_t nread;
        switch (types[col_idx]) {
          case ARROWFLOAT:   nread = op.read<arrow::FloatType>();      break;
          case ARROWDOUBLE:  nread = op.read<arrow::DoubleType>();     break;
          case ARROWINT64:   nread = op.read<arrow::Int64Type>();      break;
          case ARROWUINT64:  nread = op.read<arrow::UInt64Type>();     break;
          case ARROWBOOLEAN: nread = op.read<arrow::BooleanType>();    break;
          case ARROWINT32:   nread = op.read<arrow::Int32Type>();      break;
          case ARROWUINT32:  nread = op.read<arrow::UInt32Type>();     break;
          case ARROWDECIMAL: nread = op.read<arrow::Decimal128Type>(); break;
          default:
            // TODO we might want to have our own exception types on C++ side,
            // too 
            throw std::domain_error("Unknown Arrow type");
        }

        if (nrows_in_iter == -1) {
          // this is the first time we are reading in this row group. We'll need
          // to bump up the row_idx in the end of the iteration by nrows_read
          nrows_in_iter = nread;
        }
        else {
          // we already now how many rows we are reading per column, which is
          // stored in nrows_in_iter. But did we read the correct number of rows
          // in this particular iteration?
          if (nread != nrows_in_iter) {
            std::stringstream msgStream;
            msgStream << "Uneven number of rows are read.";
            msgStream << " Expected " << nrows_in_iter;
            msgStream << " But read " << nread << " instead\n";
            throw std::length_error(msgStream.str());
            return ARROWERROR;
          }
        }
      }

      row_idx += nrows_in_iter;
    }
    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}
} // end namespace akcpp

// Returns the number of elements read
template <typename ReaderType, typename ChplType>
int64_t readColumn(void* chpl_arr, int64_t *startIdx, std::shared_ptr<parquet::ColumnReader> column_reader,
                bool hasNonFloatNulls, int64_t i, int64_t numElems, int64_t batchSize,
                int64_t values_read, bool* where_null_chpl) {
  int16_t definition_level; // nullable type and only reading single records in batch
  auto chpl_ptr = (ChplType*)chpl_arr;
  int64_t num_read = 0;
  ReaderType* reader =
    static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  if (not hasNonFloatNulls) {
    while (reader->HasNext() && i < numElems) {
      if((numElems - i) < batchSize) // adjust batchSize if needed
        batchSize = numElems - i;
      (void)reader->ReadBatch(batchSize, nullptr, nullptr, &chpl_ptr[i], &values_read);
      i+=values_read;
      num_read += values_read;
    }
  }
  else {
    while (reader->HasNext() && i < numElems) {
      (void)reader->ReadBatch(1, &definition_level, nullptr, &chpl_ptr[i], &values_read);
      // if values_read is 0, that means that it was a null value
      if(values_read == 0) {
        where_null_chpl[i] = true;
      }
      i++;
      num_read++;
    }
  }
  return num_read;
}

template <typename ReaderType, typename ChplType, typename PqType>
int64_t readColumnDbFl(void* chpl_arr, int64_t *startIdx,
                       std::shared_ptr<parquet::ColumnReader> column_reader,
                    bool hasNonFloatNulls, int64_t i, int64_t numElems, int64_t batchSize,
                    int64_t values_read, bool* where_null_chpl) {
  int16_t definition_level; // nullable type and only reading single records in batch
  auto chpl_ptr = (ChplType*)chpl_arr;
  ReaderType* reader =
    static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  int64_t num_read = 0;
  while (reader->HasNext() && i < numElems) {
    PqType value;
    (void)reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
    // if values_read is 0, that means that it was a null value
    if(values_read > 0) {
      // this means it wasn't null
      chpl_ptr[i] = (ChplType) value;
    } else {
      chpl_ptr[i] = NAN;
    }
    i++;
    num_read++;
  }
  return num_read;
}

template <typename ReaderType, typename ChplType, typename PqType>
int64_t readColumnIrregularBitWidth(void* chpl_arr, int64_t *startIdx, std::shared_ptr<parquet::ColumnReader> column_reader,
                                 bool hasNonFloatNulls, int64_t i, int64_t numElems, int64_t batchSize,
                                 int64_t values_read, bool* where_null_chpl) {
  int16_t definition_level; // nullable type and only reading single records in batch
  auto chpl_ptr = (ChplType*)chpl_arr;
  ReaderType* reader =
    static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  int64_t num_read = 0;
  if (not hasNonFloatNulls) {
    PqType* tmpArr = (PqType*)malloc(batchSize * sizeof(PqType));
    while (reader->HasNext() && i < numElems) {
      if((numElems - i) < batchSize) // adjust batchSize if needed
        batchSize = numElems - i;

      // Can't read directly into chpl_ptr because it is int64
      (void)reader->ReadBatch(batchSize, nullptr, nullptr,
                              (int32_t*)tmpArr, &values_read);
      for (int64_t j = 0; j < values_read; j++)
        chpl_ptr[i+j] = (ChplType)tmpArr[j];
      i+=values_read;
      num_read+=values_read;
    }
    free(tmpArr);
  }
  else {
    PqType tmp;
    while (reader->HasNext() && i < numElems) {
      (void)reader->ReadBatch(1, &definition_level, nullptr,
                              (int32_t*)&tmp, &values_read);
      // if values_read is 0, that means that it was a null value
      if(values_read == 0) {
        where_null_chpl[i] = true;
      }
      else {
        chpl_ptr[i] = (int64_t)tmp;
      }
      i++;
      num_read++;
    }
  }
  return num_read;
}

int cpp_readStrColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t batchSize, char** errMsg) {
  try {
    const int64_t ty = cpp_getType(filename, colname, errMsg);
    if (ty != ARROWSTRING) {
      std::string msg = std::string("Dataset: ") + colname + " is not a string column in file: " + filename;
      *errMsg = strdup(msg.c_str());
      return ARROWERROR;
    }

    ARROW_ASSIGN_OR_RAISE(auto dataset, MakeDatasetFromPath(filename));
    const int64_t bs = (batchSize > 0) ? batchSize : 65536;
    ARROW_ASSIGN_OR_RAISE(auto reader, MakeSingleColumnReader(dataset, colname, bs));

    auto out = static_cast<uint8_t*>(chpl_arr);
    int64_t out_i = 0;

    std::shared_ptr<arrow::RecordBatch> batch;
    while (out_i < numElems) {
      ARROW_ASSIGN_OR_RAISE(batch, reader->ReadNext());
      if (!batch) break;
      const auto& arr = batch->column(0);

      // Handle both string and large_string; treat binary similarly.
      if (arr->type_id() == arrow::Type::STRING || arr->type_id() == arrow::Type::BINARY) {
        auto a = std::static_pointer_cast<arrow::BinaryArray>(arr);
        for (int64_t r = 0; r < a->length() && out_i < numElems; ++r) {
          if (a->IsNull(r)) {
            out[out_i++] = 0;
            continue;
          }
          const auto view = a->GetView(r);
          const int64_t copy_n = std::min<int64_t>(static_cast<int64_t>(view.size()), numElems - out_i - 1);
          std::memcpy(out + out_i, view.data(), static_cast<size_t>(copy_n));
          out_i += copy_n;
          out[out_i++] = 0;
        }
      } else if (arr->type_id() == arrow::Type::LARGE_STRING || arr->type_id() == arrow::Type::LARGE_BINARY) {
        auto a = std::static_pointer_cast<arrow::LargeBinaryArray>(arr);
        for (int64_t r = 0; r < a->length() && out_i < numElems; ++r) {
          if (a->IsNull(r)) {
            out[out_i++] = 0;
            continue;
          }
          const auto view = a->GetView(r);
          const int64_t copy_n = std::min<int64_t>(static_cast<int64_t>(view.size()), numElems - out_i - 1);
          std::memcpy(out + out_i, view.data(), static_cast<size_t>(copy_n));
          out_i += copy_n;
          out[out_i++] = 0;
        }
      } else {
        std::string msg = std::string("Unsupported string storage type for column ") + colname;
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }
    }
    return 0;
  } catch (const std::exception& e) {
    if (errMsg) *errMsg = strdup(e.what());
    return ARROWERROR;
  } catch (const arrow::Status& st) {
    if (errMsg) *errMsg = strdup(st.message().c_str());
    return ARROWERROR;
  }
}


int cpp_readColumnByName(const char* filename, void* chpl_arr, bool* where_null_chpl, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, int64_t byteLength, bool hasNonFloatNulls, char** errMsg) {
  try {
    const int64_t ty = cpp_getType(filename, colname, errMsg);
    ARROW_ASSIGN_OR_RAISE(auto dataset, MakeDatasetFromPath(filename));

    // Default batch size if caller passes 0/negative
    const int64_t bs = (batchSize > 0) ? batchSize : 65536;
    ARROW_ASSIGN_OR_RAISE(auto reader, MakeSingleColumnReader(dataset, colname, bs));

    int64_t out_i = 0;
    int64_t to_skip = startIdx;

    std::shared_ptr<arrow::RecordBatch> batch;
    while (out_i < numElems) {
      ARROW_ASSIGN_OR_RAISE(batch, reader->ReadNext());
      if (!batch) break;  // EOF

      const auto& arr = batch->column(0);
      const int64_t n = arr->length();

      // Skip leading rows for startIdx
      if (to_skip >= n) {
        to_skip -= n;
        continue;
      }

      const int64_t begin = to_skip;
      const int64_t avail = n - begin;
      const int64_t take = std::min<int64_t>(avail, numElems - out_i);
      to_skip = 0;

      auto write_null = [&](int64_t rel_i) {
        if (where_null_chpl) where_null_chpl[out_i + rel_i] = true;
      };
      auto clear_null = [&](int64_t rel_i) {
        if (where_null_chpl) where_null_chpl[out_i + rel_i] = false;
      };

      switch (ty) {
        case ARROWINT64: {
          auto a = std::static_pointer_cast<arrow::Int64Array>(arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); write_value<int64_t>(chpl_arr, out_i + j, 0); }
            else { clear_null(j); write_value<int64_t>(chpl_arr, out_i + j, a->Value(idx)); }
          }
          break;
        }
        case ARROWUINT64: {
          auto a = std::static_pointer_cast<arrow::UInt64Array>(arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); write_value<uint64_t>(chpl_arr, out_i + j, 0); }
            else { clear_null(j); write_value<uint64_t>(chpl_arr, out_i + j, a->Value(idx)); }
          }
          break;
        }
        case ARROWINT32: {
          auto a = std::static_pointer_cast<arrow::Int32Array>(arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); write_value<int32_t>(chpl_arr, out_i + j, 0); }
            else { clear_null(j); write_value<int32_t>(chpl_arr, out_i + j, a->Value(idx)); }
          }
          break;
        }
        case ARROWUINT32: {
          auto a = std::static_pointer_cast<arrow::UInt32Array>(arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); write_value<uint32_t>(chpl_arr, out_i + j, 0); }
            else { clear_null(j); write_value<uint32_t>(chpl_arr, out_i + j, a->Value(idx)); }
          }
          break;
        }
        case ARROWBOOLEAN: {
          auto a = std::static_pointer_cast<arrow::BooleanArray>(arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); write_value<bool>(chpl_arr, out_i + j, false); }
            else { clear_null(j); write_value<bool>(chpl_arr, out_i + j, a->Value(idx)); }
          }
          break;
        }
        case ARROWFLOAT: {
          auto a = std::static_pointer_cast<arrow::FloatArray>(arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); write_value<float>(chpl_arr, out_i + j, 0.0f); }
            else { clear_null(j); write_value<float>(chpl_arr, out_i + j, a->Value(idx)); }
          }
          break;
        }
        case ARROWDOUBLE: {
          auto a = std::static_pointer_cast<arrow::DoubleArray>(arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); write_value<double>(chpl_arr, out_i + j, 0.0); }
            else { clear_null(j); write_value<double>(chpl_arr, out_i + j, a->Value(idx)); }
          }
          break;
        }
        case ARROWTIMESTAMP: {
          // Represented as int64 in Arkouda code path.
          auto a = std::static_pointer_cast<arrow::TimestampArray>(arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); write_value<int64_t>(chpl_arr, out_i + j, 0); }
            else { clear_null(j); write_value<int64_t>(chpl_arr, out_i + j, a->Value(idx)); }
          }
          break;
        }
        case ARROWDECIMAL: {
          // Legacy behaviour: convert to double.
          auto a = std::static_pointer_cast<arrow::Decimal128Array>(arr);
          auto out = static_cast<double*>(chpl_arr);
          for (int64_t j = 0; j < take; ++j) {
            const int64_t idx = begin + j;
            if (a->IsNull(idx)) { write_null(j); out[out_i + j] = 0.0; }
            else { clear_null(j); out[out_i + j] = a->GetValue(idx).ToDouble(0); }
          }
          break;
        }
        default: {
          std::string msg = std::string("Dataset: ") + colname + " is an unsupported type in file: " + filename;
          *errMsg = strdup(msg.c_str());
          return ARROWERROR;
        }
      }

      out_i += take;
    }

    // If caller provided a null bitmap, ensure we didn't leave garbage for unread tail.
    if (where_null_chpl) {
      for (int64_t k = out_i; k < numElems; ++k) where_null_chpl[k] = true;
    }
    return 0;
  } catch (const std::exception& e) {
    if (errMsg) *errMsg = strdup(e.what());
    return ARROWERROR;
  } catch (const arrow::Status& st) {
    if (errMsg) *errMsg = strdup(st.message().c_str());
    return ARROWERROR;
  }
}

int cpp_readListColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
    if (ty == ARROWLIST){
      int64_t lty = cpp_getListType(filename, colname, errMsg);
      std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
        parquet::ParquetFileReader::OpenFile(filename, false);

      std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
      int num_row_groups = file_metadata->num_row_groups();

      auto idx = file_metadata -> schema() -> group_node() -> FieldIndex(colname);
      if(idx < 0) {
        std::string dname(colname);
        std::string fname(filename);
        std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }

      int64_t i = 0;
      int64_t arrayIdx = 0;
      for (int r = 0; r < num_row_groups; r++) {
        std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(r);

        int64_t values_read = 0;
        int16_t definition_level; // needed for any type that is nullable

        std::shared_ptr<parquet::ColumnReader> column_reader = row_group_reader->Column(idx);
        if(lty == ARROWINT64 || lty == ARROWUINT64) {
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (int64_t*)chpl_arr;
          parquet::Int64Reader* reader =
            static_cast<parquet::Int64Reader*>(column_reader.get());
          startIdx -= reader->Skip(startIdx);

          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &chpl_ptr[arrayIdx], &values_read);
            // if values_read is 0, that means that it was an empty seg
            if (values_read != 0) {
              arrayIdx++;
            }
            i++;
          }
        } else if(lty == ARROWINT32 || lty == ARROWUINT32) {
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (int64_t*)chpl_arr;
          parquet::Int32Reader* reader =
            static_cast<parquet::Int32Reader*>(column_reader.get());
          startIdx -= reader->Skip(startIdx);

          int32_t tmp;
          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &tmp, &values_read);
            // if values_read is 0, that means that it was an empty seg
            if (values_read != 0) {
              chpl_ptr[arrayIdx] = (int64_t)tmp;
              arrayIdx++;
            }
            i++;
          }
        } else if (lty == ARROWSTRING) {
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (unsigned char*)chpl_arr;
          parquet::ByteArrayReader* reader =
            static_cast<parquet::ByteArrayReader*>(column_reader.get());

          while (reader->HasNext()) {
            parquet::ByteArray value;
            (void)reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
            // if values_read is 0, that means that it was a null value
            if(values_read > 0 && definition_level == 3) {
              for(int j = 0; j < value.len; j++) {
                chpl_ptr[i] = value.ptr[j];
                i++;
              }
              i++; // skip one space so the strings are null terminated with a 0
            }
          }
        } else if(lty == ARROWBOOLEAN) {
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (bool*)chpl_arr;
          parquet::BoolReader* reader =
            static_cast<parquet::BoolReader*>(column_reader.get());
          startIdx -= reader->Skip(startIdx);

          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &chpl_ptr[arrayIdx], &values_read);
            // if values_read is 0, that means that it was an empty seg
            if (values_read != 0) {
              arrayIdx++;
            }
            i++;
          }
        } else if(lty == ARROWFLOAT) {
          // convert to simpler single batch to sidestep this seemingly architecture dependent (see issue #3234)
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (double*)chpl_arr;
          parquet::FloatReader* reader =
            static_cast<parquet::FloatReader*>(column_reader.get());

          float tmp;
          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &tmp, &values_read);
            // if values_read is 0, that means that it was a null value or empty seg
            if (values_read != 0) {
              chpl_ptr[arrayIdx] = (double) tmp;
              arrayIdx++;
            }
            else {
              // check if nan otherwise it's an empty seg
              if (definition_level == 2) {
                chpl_ptr[arrayIdx] = NAN;
                arrayIdx++;
              }
            }
            i++;
          }
        } else if(lty == ARROWDOUBLE) {
          // convert to simpler single batch to sidestep this seemingly architecture dependent (see issue #3234)
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (double*)chpl_arr;
          parquet::DoubleReader* reader =
            static_cast<parquet::DoubleReader*>(column_reader.get());

          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &chpl_ptr[arrayIdx], &values_read);
            // if values_read is 0, that means that it was a null value or empty seg
            if (values_read != 0) {
              arrayIdx++;
            }
            else {
              // check if nan otherwise it's an empty seg
              if (definition_level == 2) {
                chpl_ptr[arrayIdx] = NAN;
                arrayIdx++;
              }
            }
            i++;
          }
        }
      }
      return 0;
    }
    return ARROWERROR;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int64_t cpp_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
  try {
    const int64_t ty = cpp_getType(filename, colname, errMsg);
    if (ty != ARROWSTRING) {
      std::string msg = std::string("Dataset: ") + colname + " is not a string column in file: " + filename;
      *errMsg = strdup(msg.c_str());
      return ARROWERROR;
    }

    ARROW_ASSIGN_OR_RAISE(auto dataset, MakeDatasetFromPath(filename));
    const int64_t bs = (batchSize > 0) ? batchSize : 65536;
    ARROW_ASSIGN_OR_RAISE(auto reader, MakeSingleColumnReader(dataset, colname, bs));

    auto offsets = static_cast<int64_t*>(chpl_offsets);
    int64_t byteSize = 0;
    int64_t out_row = 0;
    int64_t to_skip = startIdx;

    std::shared_ptr<arrow::RecordBatch> batch;
    while (out_row < numElems) {
      ARROW_ASSIGN_OR_RAISE(batch, reader->ReadNext());
      if (!batch) break;

      const auto& arr = batch->column(0);
      const int64_t n = arr->length();
      if (to_skip >= n) {
        to_skip -= n;
        continue;
      }
      const int64_t begin = to_skip;
      const int64_t avail = n - begin;
      const int64_t take = std::min<int64_t>(avail, numElems - out_row);
      to_skip = 0;

      auto handle_binary = [&](auto a) {
        for (int64_t j = 0; j < take; ++j) {
          const int64_t idx = begin + j;
          if (a->IsNull(idx)) {
            offsets[out_row] = 1;
            byteSize += 1;
          } else {
            const auto v = a->GetView(idx);
            offsets[out_row] = static_cast<int64_t>(v.size()) + 1;
            byteSize += static_cast<int64_t>(v.size()) + 1;
          }
          out_row++;
        }
      };

      if (arr->type_id() == arrow::Type::STRING || arr->type_id() == arrow::Type::BINARY) {
        handle_binary(std::static_pointer_cast<arrow::BinaryArray>(arr));
      } else if (arr->type_id() == arrow::Type::LARGE_STRING || arr->type_id() == arrow::Type::LARGE_BINARY) {
        handle_binary(std::static_pointer_cast<arrow::LargeBinaryArray>(arr));
      } else {
        std::string msg = std::string("Unsupported string storage type for column ") + colname;
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }
    }

    return byteSize;
  } catch (const std::exception& e) {
    if (errMsg) *errMsg = strdup(e.what());
    return ARROWERROR;
  } catch (const arrow::Status& st) {
    if (errMsg) *errMsg = strdup(st.message().c_str());
    return ARROWERROR;
  }
}

int64_t cpp_getStringListColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
    int64_t dty; // used to store the type of data so we can handle lists
    if (ty == ARROWLIST) { // get the type of the list so we can verify it is ARROWSTRING
      dty = cpp_getListType(filename, colname, errMsg);
    }
    else {
      dty = ty;
    }
    auto offsets = (int64_t*)chpl_offsets;
    int64_t byteSize = 0;

    if(dty == ARROWSTRING) {
      std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
        parquet::ParquetFileReader::OpenFile(filename, false);

      std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
      int num_row_groups = file_metadata->num_row_groups();

      int64_t i = 0;
      for (int r = 0; r < num_row_groups; r++) {
        std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(r);

        int64_t values_read = 0;

        std::shared_ptr<parquet::ColumnReader> column_reader;

        int64_t idx;
        if (ty == ARROWLIST) {
          idx = file_metadata -> schema() -> group_node() -> FieldIndex(colname);
        } else {
          idx = file_metadata -> schema() -> ColumnIndex(colname);
        }

        if(idx < 0) {
          std::string dname(colname);
          std::string fname(filename);
          std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
          *errMsg = strdup(msg.c_str());
          return ARROWERROR;
        }
        column_reader = row_group_reader->Column(idx);

        int16_t definition_level;
        parquet::ByteArrayReader* ba_reader =
          static_cast<parquet::ByteArrayReader*>(column_reader.get());

        int64_t numRead = 0;
        while (ba_reader->HasNext() && numRead < numElems) {
          parquet::ByteArray value;
          (void)ba_reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
          if ((ty == ARROWLIST && definition_level == 3) || ty == ARROWSTRING) {
            if(values_read > 0) {
              offsets[i] = value.len + 1;
              byteSize += value.len + 1;
              numRead += values_read;
            } else {
              offsets[i] = 1;
              byteSize+=1;
              numRead+=1;
            }
            i++;
          }
        }
      }
      return byteSize;
    }
    return ARROWERROR;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int64_t cpp_getListColumnSize(const char* filename, const char* colname, void* chpl_seg_sizes, int64_t numElems, int64_t startIdx, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
    auto seg_sizes = (int64_t*)chpl_seg_sizes;
    int64_t listSize = 0;
    
    if (ty == ARROWLIST){
      int64_t lty = cpp_getListType(filename, colname, errMsg);
      std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
        parquet::ParquetFileReader::OpenFile(filename, false);

      std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
      int num_row_groups = file_metadata->num_row_groups();

      auto idx = file_metadata -> schema() -> group_node() -> FieldIndex(colname);
      if(idx < 0) {
        std::string dname(colname);
        std::string fname(filename);
        std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }

      int64_t i = 0;
      int64_t vct = 0;
      int64_t seg_size = 0;
      int64_t off = 0;
      bool first = true;
      for (int r = 0; r < num_row_groups; r++) {
        std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(r);

        int64_t values_read = 0;

        std::shared_ptr<parquet::ColumnReader> column_reader;

        column_reader = row_group_reader->Column(idx);
        int16_t definition_level;
        int16_t rep_lvl;

        if(lty == ARROWINT64 || lty == ARROWUINT64) {
          parquet::Int64Reader* int_reader =
            static_cast<parquet::Int64Reader*>(column_reader.get());

          while (int_reader->HasNext()) {
            int64_t value;
            (void)int_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if (values_read == 0 || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if (values_read != 0 && !int_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if(lty == ARROWINT32 || lty == ARROWUINT32) {
          parquet::Int32Reader* int_reader =
            static_cast<parquet::Int32Reader*>(column_reader.get());

          while (int_reader->HasNext()) {
            int32_t value;
            (void)int_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if (values_read == 0 || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if (values_read != 0 && !int_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if (lty == ARROWSTRING) {
          parquet::ByteArrayReader* reader =
            static_cast<parquet::ByteArrayReader*>(column_reader.get());

          while (reader->HasNext()) {
            parquet::ByteArray value;
            (void)reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if (values_read == 0 || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if (values_read != 0 && !reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if(lty == ARROWBOOLEAN) {
          parquet::BoolReader* bool_reader =
            static_cast<parquet::BoolReader*>(column_reader.get());

          while (bool_reader->HasNext()) {
            bool value;
            (void)bool_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if (values_read == 0 || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if (values_read != 0 && !bool_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if (lty == ARROWFLOAT) {
          parquet::FloatReader* float_reader =
            static_cast<parquet::FloatReader*>(column_reader.get());

          int64_t numRead = 0;
          while (float_reader->HasNext()) {
            float value;
            (void)float_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if ((values_read == 0 && definition_level != 2) || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0 || (values_read == 0 && definition_level == 2)) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if ((values_read != 0 || (values_read == 0 && definition_level == 2)) && !float_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if(lty == ARROWDOUBLE) {
          parquet::DoubleReader* dbl_reader =
            static_cast<parquet::DoubleReader*>(column_reader.get());

          while (dbl_reader->HasNext()) {
            double value;
            (void)dbl_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if ((values_read == 0 && definition_level != 2) || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0 || (values_read == 0 && definition_level == 2)) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if ((values_read != 0 || (values_read == 0 && definition_level == 2)) && !dbl_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        }
      }
      return vct;
    }
    return ARROWERROR;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

extern "C" {
  int c_readStrColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t batchSize, char** errMsg) {
    return cpp_readStrColumnByName(filename, chpl_arr, colname, numElems, batchSize, errMsg);
  }
  
  int c_readAllCols(const char* filename, void** chpl_arrs, int* types,
                         bool* where_null_chpl, int64_t numElems,
                         int64_t startIdx, int64_t batchSize,
                         chplEnum_t nullMode, char** errMsg) {
    return akcpp::readAllCols(filename, chpl_arrs, types, where_null_chpl,
                              numElems, startIdx, batchSize, nullMode,
                              errMsg);
  }

  int c_readColumnByName(const char* filename, void* chpl_arr,
                         bool* where_null_chpl, const char* colname,
                         int64_t numElems, int64_t startIdx, int64_t batchSize,
                         int64_t byteLength, bool hasNonFloatNulls,
                         char** errMsg) {
    return cpp_readColumnByName(filename, chpl_arr, where_null_chpl, colname,
                                numElems, startIdx, batchSize, byteLength,
                                hasNonFloatNulls, errMsg);
  }

  int c_readListColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_readListColumnByName(filename, chpl_arr, colname, numElems, startIdx, batchSize, errMsg);
  }

  int64_t c_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_getStringColumnNumBytes(filename, colname, chpl_offsets, numElems, startIdx, batchSize, errMsg);
  }

  int64_t c_getListColumnSize(const char* filename, const char* colname, void* chpl_seg_sizes, int64_t numElems, int64_t startIdx, char** errMsg) {
    return cpp_getListColumnSize(filename, colname, chpl_seg_sizes, numElems, startIdx, errMsg);
  }

  int64_t c_getStringListColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_getStringListColumnNumBytes(filename, colname, chpl_offsets, numElems, startIdx, batchSize, errMsg);
  }
}
