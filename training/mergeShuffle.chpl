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

const N = 10**3;
const K = 4;
const size = 2**K;

var x = makeDistArray(for i in 0..#N do i);

proc shuffleRange(ref x: [] ?t, lower: int, upper: int){
    coforall loc in Locales do on loc {
    
        const low = x.localSubdomain(loc=here).low;
        const high = x.localSubdomain(loc=here).high; 
        const localLower = max(low, lower);
        const localUpper = min(high, upper);
        
        forall j in localLower..localUpper with(var randStreamInt = new randomStream(int)) {
            const idx = randStreamInt.choose(j..upper);
            if (j != idx ){
                x[j] <=> x[idx];
            }
        }
    }
}

proc shuffleLocales(ref x: [] ?t){
    writeln("SHUFFLING");
    coforall loc in Locales do on loc {
        shuffleRange(x, x.localSubdomain(loc=here).low, x.localSubdomain(loc=here).high);       
    }
}

proc merge(ref x: [] ?t, s: int, n1: int, n2: int){
    var i: int = s;
    var j: int = s + n1;
    var n: int = s + n1 + n2 - 1;
    const threshold = (n1: real)/((n1 + n2):real);

    for loc in Locales do on loc {
      const low = x.localSubdomain(loc=here).low;
      const high = x.localSubdomain(loc=here).high;
      var randStream = new randomStream(real);

      while(true){
          if (i < low) | (i > high){
            break;
            }
      
          if randStream.next() < threshold {
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
    
    }
    
    shuffleRange(x, i, n);
}


proc merge2(ref x: [] ?t, s: int, n1: int, n2: int){
    var i: int = s;
    var j: int = s + n1;
    var n: int = s + n1 + n2 - 1;
    const threshold = (n1: real)/((n1 + n2):real);
    
    writeln("\nMerging....");
    writeln("i: ", i);
    writeln("j: ", j);
    writeln("n: ", n);
    writeln("threshold: ", threshold);

    for loc in Locales do on loc {
        const low = x.localSubdomain(loc=here).low;
        const high = x.localSubdomain(loc=here).high;
      
        const localLow = max(low, s);
        const localHigh = min(high, n);
      
        var randStream = new randomStream(real);

        for i in localLow..localHigh {
            writeln("i: ", i);
            writeln("j: ", j);
        
            if (j<=n) && (i < j) { 
                if(randStream.next() > threshold) {
                    x[i] <=> x[j];
                    j += 1;   
                    writeln("case1");
                    writeln("swap ", i, " with ", j);       
                    if(randStream.next() > threshold){

                        writeln("case2");
                        writeln("increment ", j);     
                        j += 1; 
                    } 
                    }
            }
        }
    }
    
    //shuffleRange(x, i, n);
}

proc getDomainLows(ref x: [] ?t){
    var domainLows: [0..#numLocales] int;
    for loc in Locales do on loc {
        domainLows[here.id] = x.localSubdomain(loc=here).low;
    }
    return domainLows;
}


proc getDomainHighs(ref x: [] ?t){
    var domainHighs: [0..#numLocales] int;
    for loc in Locales do on loc {
        domainHighs[here.id] = x.localSubdomain(loc=here).high;
    } 
    return domainHighs;
}

proc getDomainSizes(ref x: [] ?t){
    var domainSizes: [0..#numLocales] int;
    for loc in Locales do on loc {
        domainSizes[here.id] = x.localSubdomain(loc=here).size;
    }
    return domainSizes;
}


proc mergeShuffle(ref x: [] ?t){
    const numRounds = log2(numLocales) + 1;
    const domainLows = getDomainLows(x);
    const domainHighs = getDomainHighs(x);
    
    shuffleLocales(x);
  
    for m in 0..#(numRounds) {    
    //for m in 0..0 {
        const maxLocalesPerPrevChunk = 2**m;
        const numNewChunks = (numLocales - 1) / (2 * maxLocalesPerPrevChunk) + 1;

        //forall chunk in 0..#numNewChunks {
        for chunk in 0..#numNewChunks {
           writeln("\nChunk: ", chunk);
           const startLocale = chunk * 2 * maxLocalesPerPrevChunk;
           const endLocale = min(startLocale + maxLocalesPerPrevChunk - 1, numLocales - 1 );
           const startLocale2 = min(endLocale + 1, numLocales - 1 );
           const endLocale2 = min(startLocale2 + maxLocalesPerPrevChunk - 1, numLocales - 1 );
           
                //writeln("startLocale: ", startLocale);                   
                //writeln("endLocale: ", endLocale);
                //writeln("startLocale2: ", startLocale2);                    
                //writeln("endLocale2: ", endLocale2);                 
           //    Only merge if the chunks are different from eachother
           if( endLocale < startLocale2 ){
  
                const start = domainLows[startLocale];
                const size1 = domainHighs[endLocale] - domainLows[startLocale] + 1;
                const size2 = domainHighs[endLocale2] - domainLows[startLocale2] + 1;
              
                merge(x, start, size1, size2);
           }else{
                //writeln("*endLocale: ", endLocale);
               // writeln("*startLocale2: ", startLocale2);                
           }
        }
    }
}


writeln("N: ", N);
writeln("K: ", K);
writeln("size: ", size);

//writeln("x: ", x);

startCommDiagnostics();
mergeShuffle(x);
//shuffle(x);
printCommDiagnosticsTable();
stopCommDiagnostics();

writeln("x: ", x);
//writeln("x[800:999]: ", x[800:999]);


