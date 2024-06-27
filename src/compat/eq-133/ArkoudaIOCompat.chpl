module ArkoudaIOCompat {
  use IO, JSON, Set;

  proc getByteOrderCompat() throws {
    use IO;
    var writeVal = 1, readVal = 0;
    var tmpf = openMemFile();
    tmpf.writer(serializer = new binarySerializer(endian=endianness.big)).write(writeVal);
    tmpf.reader(deserializer=new binaryDeserializer(endian=endianness.native)).read(readVal);
    return if writeVal == readVal then "big" else "little";
  }

  proc fileIOReaderCompat(infile) throws {
    return infile.reader(deserializer=new binarySerializer(endian=endianness.native));
  }
}
