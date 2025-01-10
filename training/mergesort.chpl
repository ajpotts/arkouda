use Math;
use Random;
use CommDiagnostics;
public use BlockDist;
    import ChplConfig;



    /*
        Available domain maps.
    */
    enum Dmap {defaultRectangular, blockDist};

    private param defaultDmap = if ChplConfig.CHPL_COMM == "none" then Dmap.defaultRectangular
                                                                    else Dmap.blockDist;
    config param MyDmap:Dmap = defaultDmap;



    /*
        Makes a domain distributed according to :param:`MyDmap`.

        :arg shape: size of domain in each dimension
        :type shape: int
    */
    proc makeDistDom(shape: int ...?N) {
      var rngs: N*range;
      for i in 0..#N do rngs[i] = 0..#shape[i];
      const dom = {(...rngs)};

      return makeDistDom(dom);
    }

    proc makeDistDom(dom: domain(?)) {
      select MyDmap {
        when Dmap.defaultRectangular {
          return dom;
        }
        when Dmap.blockDist {
          if dom.size > 0 {
              return blockDist.createDomain(dom);
          }
          // fix the annoyance about boundingBox being empty
          else {
            return dom dmapped new blockDist(boundingBox=dom.expand(1));
          }
        }
      }
    }

    /*
        Makes an array of specified type over a distributed domain

        :arg shape: size of the domain in each dimension
        :type shape: int

        :arg etype: desired type of array
        :type etype: type

        :returns: [] ?etype
    */
    proc makeDistArray(shape: int ...?N, type etype) throws
      where N == 1
    {
      var dom = makeDistDom((...shape));
      return dom.tryCreateArray(etype);
    }

    proc makeDistArray(shape: int ...?N, type etype) throws
      where N > 1
    {
      var a: [makeDistDom((...shape))] etype;
      return a;
    }

    proc makeDistArray(in a: [?D] ?etype) throws
      where MyDmap != Dmap.defaultRectangular && a.isDefaultRectangular()
    {
        var res = makeDistArray((...D.shape), etype);
        res = a;
        return res;
    }

    proc makeDistArray(in a: [?D] ?etype) throws
      where D.rank == 1 && (MyDmap == Dmap.defaultRectangular || !a.isDefaultRectangular())
    {
      var res = D.tryCreateArray(etype);
      res = a;
      return res;
    }

    proc makeDistArray(in a: [?D] ?etype) throws
      where D.rank > 1 && (MyDmap == Dmap.defaultRectangular || !a.isDefaultRectangular())
    {
      return a;
    }

    proc makeDistArray(D: domain(?), type etype) throws
      where D.rank == 1
    {
      var res = D.tryCreateArray(etype);
      return res;
    }

    proc makeDistArray(D: domain(?), type etype) throws
      where D.rank > 1
    {
      var res: [D] etype;
      return res;
    }

    proc makeDistArray(D: domain(?), initExpr: ?t) throws
      where D.rank == 1
    {
      return D.tryCreateArray(t, initExpr);
    }

    proc makeDistArray(D: domain(?), initExpr: ?t) throws
      where D.rank > 1
    {
      var res: [D] t = initExpr;
      return res;
    }

    /*
        Returns the type of the distributed domain

        :arg size: size of domain
        :type size: int

        :returns: type
    */
    proc makeDistDomType(size: int) type {
        return makeDistDom(size).type;
    }


//#######################################################################################################################################


writeln("START");

const N = 1000;
const K = 4;
const size = 2**K;

var randStreamInt = new randomStream(int);
var randStream = new randomStream(bool);
var rands: [0..(N/size)] bool;
randStream.fill(rands);

var x = makeDistArray(for i in 0..N do i);

proc shuffleRange(ref x: [] int, lower: int, upper: int){
        for j in lower..upper{
            const idx = randStreamInt.choose(j..upper);
            x[j] <=> x[idx];
        }
}

proc shuffle(ref x: [] int){
    writeln("SHUFFLING");
    coforall loc in Locales do on loc {
        shuffleRange(x, x.localSubdomain().low, x.localSubdomain().high);       
    }
}

proc log(i: int, j: int, n: int){
    writeln("\ni: ", i);
    writeln("j: ", j);
    writeln("n: ", n);
}

proc merge(ref x: [] int, s: int, n1: int, n2: int){
    var i: int = s;
    var j: int = s + n1;
    var n: int = s + n1 + n2;
    //log(i,j,n);
    
    
    var randStream = new randomStream(bool);
    while(true){
        if randStream.next() {
            if (i==j){
                break;
            }
        } else {
            if (j==n) {
                break;
            }
            x[i] <=> x[j];
            j += 1;
        }
        i += 1;
    }

    shuffleRange(x, i, n);

}

proc getDomainLows(ref x: [] int){
    var domainLows: [0..#numLocales] int;
    for loc in Locales do on loc {
        domainLows[here.id] = x.localSubdomain(loc=here).low;
    }
    return domainLows;
}


proc getDomainHighs(ref x: [] int){
    var domainHighs: [0..#numLocales] int;
    for loc in Locales do on loc {
        domainHighs[here.id] = x.localSubdomain(loc=here).high;
    }
    return domainHighs;
}

proc getDomainSizes(ref x: [] int){
    var domainSizes: [0..#numLocales] int;
    for loc in Locales do on loc {
        domainSizes[here.id] = x.localSubdomain(loc=here).size;
    }
    return domainSizes;
}


proc mergeShuffle(ref x: [] int){

    for i in 0..(N/size){
        const start = i * size;
        const size1 = min( size , N - start);
        const size2 = min( 2 * size, N - start - size1);
        //writeln("start: ", start);
        //writeln("size1: ", size1);
        //writeln("size2: ", size2);
        merge(x, start, size1, size2);
    }
    
}

proc mergeShuffle2(ref x: [] int){
    const N = log2(numLocales) + 1;
    
    const domainLows = getDomainLows(x);
    const domainHighs = getDomainHighs(x);
    const domainSizes = getDomainSizes(x);
    
    for m in 0..N {
        const maxLocalesPerChunk = 2**m;
        writeln("\nm: ", m);
        writeln("maxLocalesPerChunk: ", maxLocalesPerChunk);
        const numChunks = (numLocales - 1) / maxLocalesPerChunk + 1;
        writeln("numChunks: ", numChunks);        
        for chunk in 0..#(maxLocalesPerChunk - 1) {
           writeln("chunk: ", chunk);
           const start_locale = 0;
           const end_locale = 0;
           const start_locale2 = 1;
           const end_locale2 = 1;
           
           const start = domainLows[start_locale];
           const end = domainHighs[end_locale];
           const size1 = domainHighs[start_locale] - domainLows[start_locale];
           const size2 = domainHighs[start_locale2] - domainLows[start_locale2];
           merge(x, start, size1, size2);
        }
    }
    
    for i in 0..(N/size){
        const start = i * size;
        const size1 = min( size , N - start);
        const size2 = min( 2 * size, N - start - size1);
        //writeln("start: ", start);
        //writeln("size1: ", size1);
        //writeln("size2: ", size2);
        merge(x, start, size1, size2);
    }
}





writeln("N: ", N);
writeln("K: ", K);
writeln("size: ", size);
writeln("rands: ", rands);
writeln("x: ", x);

startCommDiagnostics();
shuffle(x);
mergeShuffle2(x);
printCommDiagnosticsTable();
stopCommDiagnostics();
writeln("x: ", x);

