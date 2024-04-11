/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module ArraySetops
{
    use ServerConfig;
    use Logging;

    use SymArrayDmapCompat;
    use Reflection;
    use RadixSortLSD;
    use Unique;
    use Indexing;
    use AryUtil;
    use In1d;
    use CommAggregation;
    use PrivateDist;
    use Search;
    use Sort.TwoArrayRadixSort;
    use Math;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const asLogger = new Logger(logLevel, logChannel);

    // returns intersection of 2 arrays
    proc intersect1d(a: [] ?t, b: [] t, assume_unique: bool) throws {
      //if not unique, unique sort arrays then perform operation
      if (!assume_unique) {
        var a1  = uniqueSort(a, false);
        var b1  = uniqueSort(b, false);
        return intersect1dHelper(a1, b1);
      }
      return intersect1dHelper(a,b);
    }

    proc intersect1dHelper(a: [] ?t, b: [] t) throws {
      var aux = radixSortLSD_keys(concatArrays(a,b));

      // All elements except the last
      const ref head = aux[..aux.domain.high-1];

      // All elements except the first
      const ref tail = aux[aux.domain.low+1..];
      const mask = head == tail;

      return boolIndexer(head, mask);
    }
    
    // returns the exclusive-or of 2 arrays
    proc setxor1d(a: [] ?t, b: [] t, assume_unique: bool) throws {
      //if not unique, unique sort arrays then perform operation
      if (!assume_unique) {
        var a1  = uniqueSort(a, false);
        var b1  = uniqueSort(b, false);
        return  setxor1dHelper(a1, b1);
      }
      return setxor1dHelper(a,b);
    }

    // Gets xor of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts and removes all values that occur
    // more than once
    proc setxor1dHelper(a: [] ?t, b: [] t) throws {
      const aux = radixSortLSD_keys(concatArrays(a,b));
      const ref D = aux.domain;

      // Concatenate a `true` onto each end of the array
      var flag = makeDistArray(aux.size+1, bool);
      const ref fD = flag.domain;
      
      flag[fD.low] = true;
      flag[fD.low+1..fD.high-1] = aux[..D.high-1] != aux[D.low+1..];
      flag[fD.high] = true;

      var mask;
      {
        mask = sliceTail(flag) & sliceHead(flag);
      }

      var ret = boolIndexer(aux, mask);

      return ret;
    }

    // returns the set difference of 2 arrays
    proc setdiff1d(ref a: [] ?t, ref b: [] t, assume_unique: bool) throws {
      //if not unique, unique sort arrays then perform operation
      if (!assume_unique) {
        var a1  = uniqueSort(a, false);
        var b1  = uniqueSort(b, false);
        return setdiff1dHelper(a1, b1);
      }
      return setdiff1dHelper(a,b);
    }
    
    // Gets diff of 2 arrays
    // first checks membership of values in
    // fist array in second array and stores
    // as a boolean array and inverts these
    // values and returns the array indexed
    // with this inverted array
    proc setdiff1dHelper(ref a: [] ?t, ref b: [] t) throws {
        var truth = in1d(a, b, invert=true);
        var ret = boolIndexer(a, truth);
        return ret;
    }
    
    // Gets union of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts resulting array and ensures that
    // values are unique
    proc union1d(a: [] ?t, b: [] t) throws {
      var aux;
      // Artificial scope to clean up temporary arrays
      {
        aux = concatArrays(uniqueSort(a,false), uniqueSort(b,false));
      }
      return uniqueSort(aux, false);
    }

    proc mergeHelper(ref sortedIdx: [?sD] ?t, ref permutedVals: [] ?t2, const ref idx1: [?D] t, const ref idx2: [] t, const ref val1: [] t2, const ref val2: [] t2, percentTransferLimit:int = 100) throws {
      const allocSize = sD.size;
      // create refs to arrays so it's easier to swap them if we come up with a good heuristic
      //  for which causes fewer data shuffles between locales
      const ref a = idx1;
      const ref aVal = val1;
      const ref b = idx2;
      const ref bVal = val2;

      // private space is a distributed domain of size numLocales
      var segs: [PrivateSpace] int;
      // indicates if we need to pull any values from the next locale.
      // the first bool is for a and the second is for b
      var pullLocalFlag: [0..<numLocales] (bool, bool);
      // number of elements to pull local (we only need one int because only one of a and b will ever need to
      //  fetch from the next locale). proofish:
      //  a and b are sorted, so a_max_loc_i <= a_min_loc_(i+1) and b_max_loc_i <= b_min_loc_(i+1)
      //  assume fetch condition a_min_loc_(i+1) < b_max_loc_i is met, combining these inequalities gives
      //  a_max_loc_i <= a_min_loc_(i+1) < b_max_loc_i <= b_min_loc_(i+1) => the other fetch condition cannot be met
      var pullLocalCount: [0..<numLocales] int;

      // if the merge based workflow seems too comm heavy, fall back to radix sort
      var toGiveUp = false;

      var aMin = -1, aMax = -1, bMin = -1, bMax = -1;
      // we iterate over locales serially because the logic relies on the mins and maxs referring to the previous locale
      for loc in Locales {
        on loc {  // perform loop body computation on locale #loc
          // domains of a and b that are live on this locale
          const aDom = a.localSubdomain();
          const bDom = b.localSubdomain();
          segs[here.id] = aDom.size + bDom.size;
          aMin = a[aDom.first];
          bMin = b[bDom.first];

          if loc != Locales[0] {
            // each locale determines if it needs to send values to the previous,
            //  so locale0 can skip this check (this also means indexing at [here.id - 1] is safe)

            // we've updated mins but not maxs. So mins refer to locale_i and maxs refer to locale_(i-1).
            // The previous locale needs to pull data local if it's max for one array is less
            //  than our min for the other array.
            if aMin < bMax {
              pullLocalFlag[here.id - 1] = (true, false);
              // binary search local chunk of a to find where b_min_loc_(i-1) falls,
              //  so we know how many vals to send to loc_(i-1)
              var (status, binSearchIdx) = search(a.localSlice[aDom], bMax, sorted=true);
              if (binSearchIdx == (aDom.last+1)) {
                toGiveUp = true;
              }
              // the number of elements the previous locale will need to fetch from here
              pullLocalCount[here.id - 1] = (binSearchIdx-aDom.first);
            }
            else if bMin < aMax {
              pullLocalFlag[here.id - 1] = (false, true);
              // binary search local chunk of b to find where a_min_loc_(i-1) falls,
              //  so we know how many vals to send to loc_(i-1)
              var (status, binSearchIdx) = search(b.localSlice[bDom], aMax, sorted=true);
              if (binSearchIdx == (bDom.last+1)) {
                toGiveUp = true;
              }
              // the number of elements the previous locale will need to fetch from here
              pullLocalCount[here.id - 1] = (binSearchIdx-bDom.first);
            }
          }
          aMax = a[aDom.last];
          bMax = b[bDom.last];
        }
        // break is not allowed in an on statement
        if toGiveUp {
          break;
        }
      }
      const totNumElemsMoved = + reduce pullLocalCount;
      const percentTransfered = (totNumElemsMoved:real / allocSize)*100:int;
      asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "Total number of elements moved to a different locale = %i".doFormat(totNumElemsMoved));
      asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "Percent of elements moved to a different locale = %i%%".doFormat(percentTransfered));

      if toGiveUp || (percentTransfered > percentTransferLimit) {
        // fall back to sort
        if toGiveUp {
          asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "Falling back to sort workflow since merge would need to shift an entire locale of data");
        }
        else {
          asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "Falling back to sort workflow since percent of elements moved to a different locale = %i%% exceeds percentTransferLimit = %i%%".doFormat(percentTransfered, percentTransferLimit));
        }
        sortHelper(sortedIdx, permutedVals, idx1, idx2, val1, val2);
      }
      else {
        // we met the necessary conditions to continue with merge based workflow
        segs = (+ scan segs) - segs;
        
        // give each locale constant reference to distributed arrays that won't be edited
        // a ref to a distributed array that will be edited (but only ever one index at a time so no race conditions)
        coforall loc in Locales with (const ref a, const ref b, const ref segs, ref sortedIdx, ref permutedVals) {
          // if the chpl team implements parallel merge, that would be helpful here
          // cause we could just use their merge step instead of concatenating and doing twoArrayRadixSort
          on loc {
            // localized const copies of small arrays
            const localizedPullLocalFlag = pullLocalFlag;
            const localizedPullLocalCount = pullLocalCount;
            const aDom = a.localSubdomain();
            const bDom = b.localSubdomain();

            // the ternaries allow me to declare the next variables as const
            // info for pushing to previous locale
            const numSendBack = if loc == Locales[0] then 0 else localizedPullLocalCount[here.id - 1];
            const sendBack = if loc == Locales[0] then (false, false) else localizedPullLocalFlag[here.id - 1];
            // info for fetching from next locale
            const numPullLocal = localizedPullLocalCount[here.id];
            const pullLocal = pullLocalFlag[here.id];

            // skip any indices that were pulled to previous locale
            const aLow = if sendBack[0] then (aDom.first + numSendBack) else aDom.first;
            const bLow = if sendBack[1] then (bDom.first + numSendBack) else bDom.first;
            const aSlice = aLow..<(aDom.last + 1);
            const bSlice = bLow..<(bDom.last + 1);

            var tmp: [0..<(aDom.size + bDom.size - numSendBack + numPullLocal)] (t, t2);

            // write the local part of a (minus any skipped values) into tmp
            tmp[0..<aSlice.size] = [(key,val) in zip(a.localSlice[aSlice], aVal.localSlice[aSlice])] (key,val);
            // write the local part of b (minus any skipped values) into tmp
            tmp[aSlice.size..#bSlice.size] = [(key,val) in zip(b.localSlice[bSlice], bVal.localSlice[bSlice])] (key,val);

            // add any indices that we need to pull to this locale
            // this not a local slice, but it should (hopefully) take advantage of bulk transfer
            if pullLocal[0] {
              const pullFromASlice = (aDom.last+1)..#numPullLocal;
              tmp[(aSlice.size+bSlice.size)..#(numPullLocal+1)] = [(key,val) in zip(a[pullFromASlice], aVal[pullFromASlice])] (key,val);
            }
            else if pullLocal[1] {
              const pullFromBSlice = (bDom.last+1)..#numPullLocal;
              tmp[(aSlice.size+bSlice.size)..#(numPullLocal+1)] = [(key,val) in zip(b[pullFromBSlice], bVal[pullFromBSlice])] (key,val);
            }

            // run chpl's builtin parallel radix sort for local arrays with a comparator defined in
            // RadixSortLSD.chpl which sorts on the key
            twoArrayRadixSort(tmp, new KeysRanksComparator());

            const writeToResultSlice = (segs[here.id]+numSendBack)..#tmp.size;
            sortedIdx[writeToResultSlice] = [(key, val) in tmp] key;
            permutedVals[writeToResultSlice] = [(key, val) in tmp] val;
          }
        }
      }
    }

    proc sortHelper(ref sortedIdx: [?sD] ?t, ref permutedVals: [] ?t2, const ref idx1: [?D] t, const ref idx2: [] t, const ref val1: [] t2, const ref val2: [] t2) throws {
      const allocSize = sD.size;
      var perm = makeDistArray(allocSize, int);
      forall (s, p, sp) in zip(sortedIdx, perm, radixSortLSD(concatArrays(idx1, idx2, ordered=false))) {
        (s, p) = sp;
      }
      // concatenate the values with the same ordering as the indices
      const vals = concatArrays(val1, val2, ordered=false);
      forall (p, i) in zip(permutedVals, perm) with (var agg = newSrcAggregator(t2)) {
        agg.copy(p, vals[i]);
      }
    }

    proc combineHelper(const ref idx1: [?D] ?t, const ref idx2: [] t, const ref val1: [] ?t2, const ref val2: [] t2, doMerge = false, percentTransferLimit:int = 100) throws {
      // combine two sorted lists of indices and apply the sort permutation to their associated values
      const allocSize = idx1.size + idx2.size;
      var sortedIdx = makeDistArray(allocSize, t);
      var permutedVals = makeDistArray(allocSize, t2);
      if doMerge {
        // attempt to use merge workflow. if certain conditions are met, fall back to radixsort
        mergeHelper(sortedIdx, permutedVals, idx1, idx2, val1, val2, percentTransferLimit);
      }
      else {
        sortHelper(sortedIdx, permutedVals, idx1, idx2, val1, val2);
      }
      return (sortedIdx, permutedVals);
    }

    proc sparseSumHelper(const ref idx1: [] ?t, const ref idx2: [] t, const ref val1: [] ?t2, const ref val2: [] t2, doMerge = false, percentTransferLimit:int = 100) throws {
      const (sortedIdx, permutedVals) = combineHelper(idx1, idx2, val1, val2, doMerge, percentTransferLimit);
      const sD = sortedIdx.domain;
      var firstOccurence = makeDistArray(sD, bool);
      firstOccurence[0] = true;
      forall (f, s, i) in zip(firstOccurence, sortedIdx, sD) {
        if i > sD.low {
          // most of the time sortedIdx[i-1] should be local since we are block distributed,
          // so we only have to fetch at locale boundaries
          f = (sortedIdx[i-1] != s);
        }
      }
      const numUnique = + reduce firstOccurence;
      // we have to do a first pass through data to calculate the size of the return array
      var uIdx = makeDistArray(numUnique, t);
      var summedVals = makeDistArray(numUnique, t2);
      const retIdx = + scan firstOccurence - firstOccurence;
      forall (s, p, i, f, rIdx) in zip(sortedIdx, permutedVals, sD, firstOccurence, retIdx) with (var idxAgg = newDstAggregator(t),
                                                                                            var valAgg = newDstAggregator(t2)) {
        if f {  // skip if we are not the first occurence
          idxAgg.copy(uIdx[rIdx], s);
          if i == sD.high || sortedIdx[i+1] != s {
            valAgg.copy(summedVals[rIdx], p);
          }
          else {
            // i'd like to do aggregation but I think it's possible for remote-to-remote aggregation?
            // i.e. valAgg.copy(summedVals[rIdx], p + permutedVals[i+1]);
            // this only happens during idx collisions, so it's not the most common case
            summedVals[rIdx] = p + permutedVals[i+1];
          }
        }
      }
      return (uIdx, summedVals);
    }

    proc sparseSumPartitionHelper(const ref idx1: [] ?t, const ref idx2: [] t, const ref val1: [] ?t2, const ref val2: [] t2) throws {

      const (sortedIdx, permutedVals) = mergePartitionHelper(idx1, idx2, val1, val2);
      const sD = sortedIdx.domain;
      var firstOccurence = makeDistArray(sD, bool);
      firstOccurence[0] = true;
      forall (f, s, i) in zip(firstOccurence, sortedIdx, sD) {
        if i > sD.low {
          // most of the time sortedIdx[i-1] should be local since we are block distributed,
          // so we only have to fetch at locale boundaries
          f = (sortedIdx[i-1] != s);
        }
      }
      const numUnique = + reduce firstOccurence;
      // we have to do a first pass through data to calculate the size of the return array
      var uIdx = makeDistArray(numUnique, t);
      var summedVals = makeDistArray(numUnique, t2);
      const retIdx = + scan firstOccurence - firstOccurence;
      forall (s, p, i, f, rIdx) in zip(sortedIdx, permutedVals, sD, firstOccurence, retIdx) with (var idxAgg = newDstAggregator(t),
                                                                                            var valAgg = newDstAggregator(t2)) {
        if f {  // skip if we are not the first occurence
          idxAgg.copy(uIdx[rIdx], s);
          if i == sD.high || sortedIdx[i+1] != s {
            valAgg.copy(summedVals[rIdx], p);
          }
          else {
            // i'd like to do aggregation but I think it's possible for remote-to-remote aggregation?
            // i.e. valAgg.copy(summedVals[rIdx], p + permutedVals[i+1]);
            // this only happens during idx collisions, so it's not the most common case
            summedVals[rIdx] = p + permutedVals[i+1];
          }
        }
      }
      return (uIdx, summedVals);
    }

    class partitionHelperTable{
      const DEBUG: bool = true;
      // Allocate arrays representing a table of statistics.
      // Thes statistics represent a value ranges used to chunk up the data in a and b.
      // The maximum table size will be 5 * numLocales.

      // The number of values in the statistics table.
      // There will be at least 4 * numLocales because we include the max and min of each array on each locale.
      var len: int = 4 * numLocales;
      const D = 0..< (5 * numLocales);
      type t;
      var values: [D] t;
      // Whether the statistics have been computed for this value using a.
      var aComputed: [D] bool;
      // Which locale supports the value in array a.
      var aLocId: [D] int;
      // Which index locates the value in array a, or the insertion location for the value in a.
      var aIndex: [D] int;
      // The number of elements in a that follow in the range of this value and the subsequent value.
      var aSize: [D] int;
      // Corresponding statistics for b....
      var bComputed: [D] bool;
      var bLocId: [D] int;
      var bIndex: [D] int;
      var bSize: [D] int;
      // For the return arrays, which locale should support the data in this value range.
      var returnLocId: [D] int;
      // Whether the chunk of data will need to be split because it crosses two locales.
      var needsSplit: [D] bool;
      // The number of elements from this chunk to be written to the return arrays.
      var returnSize: [D] int;

      proc init(type t) {
        this.t = t;
      }

      // Permute the array arr by the permutation perm, only permuting the first len elements.
      proc permuteInPlace(arr: [?D] ?t, perm : [?D2] int, len : int) {
        // Only permute the first len values:
        const tmp: [0..<len] t  = arr[0..<len];
        return tmp[perm];
      }

      proc sortAllInPlace(){
        var tmp: [0..<len] (t, int) = [(key,val) in zip(values[0..<len], 0..<len)] (key, val);
        twoArrayRadixSort(tmp, new KeysRanksComparator());
        const perm: [0..<len] int = [(v,i) in tmp] i;

        this.values[0..<len] = permuteInPlace(values, perm, len);
        this.aComputed[0..<len] = permuteInPlace(aComputed, perm, len);
        this.aLocId[0..<len] = permuteInPlace(aLocId, perm, len);
        this.aIndex[0..<len] = permuteInPlace(aIndex, perm, len);
        this.aSize[0..<len] = permuteInPlace(aSize, perm, len);
        this.bComputed[0..<len] = permuteInPlace(bComputed, perm, len);
        this.bLocId[0..<len] = permuteInPlace(bLocId, perm, len);
        this.bIndex[0..<len] = permuteInPlace(bIndex, perm, len);
        this.bSize[0..<len] = permuteInPlace(bSize, perm, len);
        this.returnLocId[0..<len] = permuteInPlace(returnLocId, perm, len);
        this.needsSplit[0..<len] = permuteInPlace(needsSplit, perm, len);
        this.returnSize[0..<len] = permuteInPlace(returnSize, perm, len);

        return perm;
      }

      proc sortAndUpdateStats(){
        this.sortAllInPlace();

        // Replace any -1 in aLocId and bLocId:
        // Since the arrays are sorted by value order, and the indices were assumed pre-sorted,
        // the locales will be in increasing order.  Plus, the locales for the max and min of each locale
        // are computed.  So, any missing locale values will be fall between computed values, and therefore
        // can be imputed.
        this.aLocId = max scan this.aLocId;
        this.bLocId = max scan this.bLocId;

        // Update sizes for each value range.
        forall i in 0..<len{
          if(!aComputed[i]){
            // If the index was not computed, the value falls between locales in a.
            // If the index was on a locale it would have been computed by the binary search above.
            aSize[i] = 0;
          }else if((i+1 >= len) || !aComputed[i+1]){
            // If the following value was not computed, it falls between locales. The value is the max of a locale.
            aSize[i] = 1;
          }else if(aLocId[i] != aLocId[i+1]){
            // If the locale is different than the following locale, this is the max of a locale.
            aSize[i] = 1;
          }else{
            aSize[i] = aIndex[i+1] - aIndex[i];
          }

          if(!bComputed[i]){
            bSize[i] = 0;
          }else if((i+1 >= len) || !bComputed[i+1]){
            bSize[i] = 1;
          }else if(bLocId[i] != bLocId[i+1]){
            bSize[i] = 1;
          }else{
            bSize[i] = bIndex[i+1] - bIndex[i];
          }
        }
      }

      // Compute the locale for the chunk in the return arrays
      proc updateRetLocales(const ref a: [?D] ?t, const ref b: [] t, segs:[PrivateSpace] int){
        var sum: int = 0;
        var retLoc: int = 0;

        for i in 0..<len{

          const size : int = aSize[i] + bSize[i];

          if((sum + size) <= segs[retLoc]){ // Data fits on current retLoc
            returnLocId[i] = retLoc;
            returnSize[i] = size;
            sum += size;
          }else if (( (sum + size) == segs[retLoc] + 1) && (a[aIndex[i]] == b[bIndex[i]])){
            // Data is only one too big for current retLoc.
            // Send to current locale, and allow the size to be one greater than expected.
            returnLocId[i] = retLoc;
            returnSize[i] = size;
            sum += size;
          }else if(sum < segs[retLoc]){
            //  The data is to big for current locale, and needs to be split.
            // Record the local as the current locale, since some of the data will fit here.
            returnLocId[i] = retLoc;
            // Record the size as the amount of the array that fits on the current locale.
            returnSize[i] = segs[retLoc] - sum;
            needsSplit[i] = true;
            // Reset sum to the amount of the chunk that did not fit on this locale.
            sum = size - (segs[retLoc] - sum);
            retLoc += 1;
          }else{
            // The data needs to be sent to the next locale.
            retLoc += 1;
            returnLocId[i] = retLoc;
            returnSize[i] = size;
            sum = size;
          }
        }
      }

      proc setValue(i: int, val:t){
        values[i] = val;
      }

      proc getValue(i: int){
        return this.values[i];
      }

      proc setAComputed(i: int, val: bool){
        this.aComputed[i] = val;
      }

      proc getAComputed(i: int){
        return this.aComputed[i];
      }

      proc setALocId(i: int, val: int){
        this.aLocId[i] = val;
      }

      proc getALocId(i: int){
        return this.aLocId[i];
      }

      proc setAIndex(i: int, val: int){
        this.aIndex[i] = val;
      }

      proc getAIndex(i: int){
        return this.aIndex[i];
      }

      proc setASize(i: int, val: int){
        this.aSize[i] = val;
      }

      proc getASize(i: int){
        return this.aSize[i];
      }

      proc setBComputed(i: int, val: bool){
        this.bComputed[i] = val;
      }

      proc getBComputed(i: int){
        return this.bComputed[i];
      }

      proc setBLocId(i: int, val: int){
        this.bLocId[i] = val;
      }

      proc getBLocId(i: int){
        return this.bLocId[i];
      }

      proc setBIndex(i: int, val: int){
        this.bIndex[i] = val;
      }

      proc getBIndex(i: int){
        return this.bIndex[i];
      }

      proc setBSize(i: int, val: int){
        this.bSize[i] = val;
      }

      proc getBSize(i: int){
        return this.bSize[i];
      }

      proc setReturnLocId(i: int, val: int){
        this.returnLocId[i] = val;
      }

      proc getReturnLocId(i: int){
        return this.returnLocId[i];
      }

      proc setNeedsSplit(i: int, val: int){
        this.needsSplit[i] = val;
      }

      proc getNeedsSplit(i: int){
        return this.needsSplit[i];
      }

      proc setReturnSize(i: int, val: int){
        this.returnSize[i] = val;
      }

      proc getReturnSize(i: int){
        return this.returnSize[i];
      }

      proc writeFormatted(arry:[D]) throws{
        for i in 0..<this.len{
          write("|%{#####}".format(arry[i]));
        }
        write("\n");
      }

      proc debugReturnSizes(){
        for i in 0..#(this.len-1){
          if(){
            writeln("WARNING: Return Size does not match at position ",i)
            writeln("return size: ", this.returnSize[i], " != ", this.aSize[i], " + ", this.bSize);
          }
        }
      }

      proc writeDebugStatements() throws{
        writeln("\n\n\nDebug\n\n");

        writeln("i:");
        for i in 1..<this.len{
          write("|%{#####}".format(i));
        }
        write("\n");

        writeln("values");
        this.writeFormatted(this.values);
        
        writeln("aComputed");
        this.writeFormatted(this.aComputed);

        writeln("aLocId");
        this.writeFormatted(this.aLocId);
        
        writeln("aIndex");
        this.writeFormatted(this.aIndex);

        writeln("aSize");
        this.writeFormatted(this.aSize);

        writeln("bComputed");
        this.writeFormatted(this.bComputed);

        writeln("bLocId");
        this.writeFormatted(this.values);

        writeln("bIndex");
        this.writeFormatted(this.bIndex);

        writeln("bSize");
        this.writeFormatted(this.bSize);
        
        writeln("returnLocId");
        this.writeFormatted(this.returnLocId);
        
        writeln("needsSplit");
        this.writeFormatted(this.needsSplit);
        
        writeln("returnSize");
        this.writeFormatted(this.returnSize);

        this.debugReturnSizes();
      }


    }

    proc mergePartitionHelper(const ref idx1: [?D] ?t, const ref idx2: [] t, const ref val1: [] ?t2, const ref val2: [] t2) throws {
      // combine two sorted lists of indices and apply the sort permutation to their associated values
      const allocSize = idx1.size + idx2.size;
      var returnIdx = makeDistArray(allocSize, t);
      var returnVals = makeDistArray(allocSize, t2);

      // create refs to arrays so it's easier to swap them if we come up with a good heuristic

      // for which causes fewer data shuffles between locales
      const ref a = idx1;
      const ref aVal = val1;
      const ref b = idx2;
      const ref bVal = val2;

      var table = new partitionHelperTable(t);

      // private space is a distributed domain of size numLocales
      var segs: [PrivateSpace] int;

      // Fill the stats table with max and min values from each locale.
      coforall loc in Locales with (const ref a, const ref b, ref table) {
        on loc {
          // perform loop body computation on locale #loc
          // domains of a and b that are live on this locale
          const aDom = a.localSubdomain();
          const bDom = b.localSubdomain();

          segs[here.id] = aDom.size + bDom.size;
          const aMin = a[aDom.first];
          const bMin = b[bDom.first];
          const aMax = a[aDom.last];
          const bMax = b[bDom.last];

          var idx: int = 4 * here.id;
          table.setValue(idx, aMin);
          table.setALocId(idx, here.id);
          table.setAIndex(idx, aDom.first);
          table.setAComputed(idx, true);

          idx += 1;
          table.setValue(idx, bMin);
          table.setBLocId(idx, here.id);
          table.setBIndex(idx, bDom.first);
          table.setBComputed(idx, true);

          idx += 1;
          table.setValue(idx, aMax);
          table.setALocId(idx, here.id);
          table.setAIndex(idx, aDom.last);
          table.setAComputed(idx, true);

          idx += 1;
          table.setValue(idx, bMax);
          table.setBLocId(idx, here.id);
          table.setBIndex(idx, bDom.last);
          table.setBComputed(idx, true);

        }
      }

      table.writeDebugStatements();
      table.sortAllInPlace();
      table.writeDebugStatements();

      //  Some indices will need to be computed using a binary search.
      //  Loop over the locales for this.
      coforall loc in Locales with (const ref a, const ref b, ref table){
        on loc {
          const aDom = a.localSubdomain();
          const bDom = b.localSubdomain();

          const aMin = a[aDom.first];
          const bMin = b[bDom.first];
          const aMax = a[aDom.last];
          const bMax = b[bDom.last];

          forall i in 0..<table.len{
            if(!table.getAComputed(i)){
              const valueToFind: t = table.getValue(i);
              if(valueToFind == aMin){
                table.setALocId(i, here.id);
                table.setAIndex(i, aDom.first);
                table.setAComputed(i, true);
              }else if(valueToFind == aMax){
                table.setALocId(i, here.id);
                table.setAIndex(i, aDom.last);
                table.setAComputed(i, true);
              }else if(valueToFind > aMin && valueToFind < aMax){
                const (status, binSearchIdx) = search(a.localSlice[aDom], valueToFind, sorted=true);
                table.setALocId(i, here.id);
                table.setAIndex(i, binSearchIdx);
                table.setAComputed(i, true);
              }
            }
            if(!table.getBComputed(i)){
              const valueToFind: t = table.getValue(i);
              if(valueToFind == bMin){
                table.setBLocId(i, here.id);
                table.setBIndex(i, bDom.first);
                table.setBComputed(i, true);
              }else if(valueToFind == bMax){
                table.setBLocId(i, here.id);
                table.setBIndex(i, bDom.last);
                table.setBComputed(i, true);
              }else if(valueToFind > bMin && valueToFind < bMax){
                const (status, binSearchIdx) = search(b.localSlice[bDom], valueToFind, sorted=true);
                table.setBLocId(i, here.id);
                table.setBIndex(i, binSearchIdx);
                table.setBComputed(i, true);
              }
            }
          }
        }
      }

      table.sortAndUpdateStats();
      table.writeDebugStatements();
      table.updateRetLocales(a, b, segs);

      //  Find the location to split off the min K of the two arrays.
      //  Array 2 will have fewer lookups.
      proc findMinKLocations(k: int, const ref arry1: [?D] ?t, const ref arry2: [] t, low1:  int, high1:  int, low2:  int, high2: int): (int, int, t){

        const maxIterations : int = floor(log(k)):int;

        var rangeLow1: int = low1;
        var rangeHigh1 : int = high1;

        var guessIndex1: int = floor((high1 + low1)/2):int;
        var guessIndex2: int = low2 + k - (guessIndex1  - low1) -1;
        var val1: int = arry1[guessIndex1];
        var val2: int = arry2[guessIndex2];

        proc update(){ // TODO Should range be over value2
          if( val1 > val2){
            rangeHigh1 = guessIndex1;
          } else{
            rangeLow1 = guessIndex1;
          }

          guessIndex1 = floor((rangeHigh1 + rangeLow1)/2): int ;
          guessIndex2 = low2 + k - (guessIndex1  - low1) - 1;

          val1 = arry1[guessIndex1];
          val2 = arry2[guessIndex2];
        }

        for i in 0..maxIterations{
          //  The search is over when val1 == val2,
          //  or arry1[i] = val1 > val2 >= arry1[i-1]
          //  or arry1[i] = val1 < val2 <= arry1[i+1]
          if(val1 == val2){
            return (guessIndex1, guessIndex2, max(val1, val2));
          }if(val1 > val2 && val2 >= arry1[guessIndex1-1]){

            return (guessIndex1, guessIndex2, max(val1, val2));
          }if(val1 < val2 && val2 <= arry1[guessIndex1+1]){
            return (guessIndex1, guessIndex2, max(val1, val2));
          }else{      
            update();
          }
        }
        // The problem is that this need to return both values, max and min.
        return (guessIndex1, guessIndex2, min(val1, val2));
      }

      // Writing to release sync variables allows 
      var release: sync int; // barrier release
      begin release.writeEF(0);

      //  Determine split points for cases when the segment needs to be divided between locales.
      coforall loc in Locales with (const ref a, const ref b, ref table){
        on loc {
          // len can be incremented but we only need to loop over the table entries that are already defined.
          const startingLen: int = table.len;

          forall i in 0..<startingLen with (ref table){
            if(table.getNeedsSplit(i) == true ){

              const k: int = table.getReturnSize(i);
              const aSz: int = table.getASize(i);
              const bSz: int = table.getBSize(i);

              if (table.getALocId(i) == here.id){
                if(bSz == 0){

                  const aIdx: int = table.getAIndex(i) + k;

                  var int_sync = release.readFE();
                  writeln("\nwrite case 1: ", int_sync, " on ", here.id);

                  table.setValue(table.len, a[aIdx]);
                  table.setAComputed(table.len, true);
                  table.setALocId(table.len, here.id);
                  table.setAIndex(table.len, aIdx);
                  table.setBComputed(table.len, true);
                  table.setBLocId(table.len, -1);
                  table.setBIndex(table.len, table.getBIndex(i));
                  table.len += 1;

                  release.writeEF(int_sync + 1);
                }else if(aSz > bSz) {

                  const aLow : int = table.getAIndex(i);
                  const aHigh : int = min(aLow + aSz - 1, aLow + k, aLow);

                  const bLow : int = table.getBIndex(i);
                  const bHigh : int = min(bLow + bSz - 1, bLow + k, bLow);
  
                  const (aSplitIndex,bSplitIndex, splitVal): (int, int, t) = findMinKLocations(k, a, b,  aLow, aHigh, bLow, bHigh);

                  var int_sync = release.readFE();
                  writeln("\nwrite case 2: ", int_sync, " on ", here.id);

                  table.setValue(table.len, splitVal);
                  table.setAComputed(table.len, true);
                  table.setALocId(table.len, here.id);
                  table.setAIndex(table.len, aSplitIndex);
                  table.setBComputed(table.len, true);
                  table.setBLocId(table.len, -1);
                  table.setBIndex(table.len, bSplitIndex);

                  table.len += 1;

                  release.writeEF(int_sync + 1);
                }
              }else if(table.getBLocId(i) == here.id){
                if(aSz == 0){
                  const bIdx: int = table.getBIndex(i) + k;

                  var int_sync = release.readFE();
                  writeln("\nwrite case 3: ", int_sync, " on ", here.id);

                  table.setValue(table.len, b[bIdx]);
                  table.setAComputed(table.len, true);
                  table.setALocId(table.len, -1);
                  table.setAIndex(table.len, table.getAIndex(i));
                  table.setBComputed(table.len, true);
                  table.setBLocId(table.len, here.id);
                  table.setBIndex(table.len, bIdx);
                  table.len += 1;

                  release.writeEF(int_sync + 1);
                }else if( bSz >= aSz) {

                  const aLow : int = table.getAIndex(i);
                  const aHigh : int = min(aLow + aSz - 1, aLow + k);

                  const bLow : int = table.getBIndex(i);
                  const bHigh : int = min(bLow + bSz - 1, bLow + k);

                  const (bSplitIndex, aSplitIndex, splitVal): (int, int, t) = findMinKLocations(k, b, a,  bLow, bHigh, aLow, aHigh);

                  var int_sync = release.readFE();
                  writeln("\nwrite case 4: ", int_sync, " on ", here.id);

                  table.setValue(table.len, splitVal);
                  table.setAComputed(table.len, true);
                  table.setALocId(table.len, -1);
                  table.setAIndex(table.len, aSplitIndex);
                  table.setBComputed(table.len, true);
                  table.setBLocId(table.len, here.id);
                  table.setBIndex(table.len, bSplitIndex);
                  table.len += 1;

                  release.writeEF(int_sync + 1);
                }
              }
            }
          }
        }
      }

      table.sortAndUpdateStats();

      //  Because we send ties to the same locale, segs can be off by a small amount and needs to be recalculated.
      segs = 0;
      for i in 0..table.len{
        segs[table.getReturnLocId(i)] += table.getASize(i) + table.getBSize(i);
      }

      table.updateRetLocales(a, b, segs);
      table.writeDebugStatements();

      table.returnSize = table.aSize + table.bSize;



      // The other potential problem is that segs is updated and needs to be used to recalcuate returnSize, but in an efficient way that doesn't needlessly sort
      var aSegStarts : [D] int = (+ scan table.returnSize) - table.returnSize;
      var bSegStarts : [D] int = aSegStarts + table.aSize;

      writeln("aSegStarts");
      writeln(aSegStarts);
      writeln("bSegStarts");
      writeln(bSegStarts);



      for loc in Locales{//} with (const ref a, const ref b, ref table) {
        on loc {

          const returnIdxDom = returnIdx.localSubdomain();

          for i in 0..<table.len{
            if((table.getReturnLocId(i) == here.id)){

              const aStartIndex = table.getAIndex(i);
              const aSegSize = table.getASize(i);

              const bStartIndex = table.getBIndex(i);
              const bSegSize = table.getBSize(i);

              //  Only sort if both a and b contribute some values.
              //  Otherwise, no sort is needed b/c the input indices are assumed sorted.
              if((aSegSize > 0) && (bSegSize > 0)){
                const tmpSize : int = aSegSize + bSegSize;
                var tmp: [0..#tmpSize] (t, t2);

                tmp[0..#aSegSize] = [(key,val) in zip(a[aStartIndex..#aSegSize], aVal[aStartIndex..#aSegSize])] (key,val);
                tmp[aSegSize..#bSegSize] = [(key,val) in zip(b[bStartIndex..#bSegSize], bVal[bStartIndex..#bSegSize])] (key,val);

                twoArrayRadixSort(tmp, new KeysRanksComparator());

                const writeToResultSlice = aSegStarts[i]..#tmpSize;
                returnIdx[writeToResultSlice] = [(key, val) in tmp] key;
                returnVals[writeToResultSlice] = [(key, val) in tmp] val;
                writeln("i");
                writeln(i);
                writeln("writeToResultSlice");
                writeln(writeToResultSlice);               
                writeln("returnIdx[writeToResultSlice]");
                writeln(returnIdx[writeToResultSlice]);  
              }else if(aSegSize > 0){
                const writeAToResultSlice = aSegStarts[i]..#aSegSize;
                returnIdx[writeAToResultSlice] = a[aStartIndex..#aSegSize];
                returnVals[writeAToResultSlice] = aVal[aStartIndex..#aSegSize];
                writeln("i");
                writeln(i);
                writeln("writeAToResultSlice");
                writeln(writeAToResultSlice);               
                writeln("returnIdx[writeAToResultSlice]");
                writeln(returnIdx[writeAToResultSlice]);  
              }else if(bSegSize > 0){
                const writeBToResultSlice = bSegStarts[i]..#bSegSize;
                returnIdx[writeBToResultSlice] = b[bStartIndex..#bSegSize];
                returnVals[writeBToResultSlice] = bVal[bStartIndex..#bSegSize];
                writeln("i");
                writeln(i);
                writeln("writeBToResultSlice");
                writeln(writeBToResultSlice);               
                writeln("returnIdx[writeBToResultSlice]");
                writeln(returnIdx[writeBToResultSlice]);  
              }
            }
          }
        }
      }
      return (returnIdx, returnVals);
    }
}
