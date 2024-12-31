
use Math;
use Random;




writeln("START");

const N = 100;
const K = 4;
const size = 2**K;

var randStreamInt = new randomStream(int);
var randStream = new randomStream(bool);
var rands: [0..(N/size)] bool;
randStream.fill(rands);


var x: [0..N] int;

for i in 0..N {
    x[i] = i;
}

proc shuffleRange(ref x: [] int, lower: int, upper: int){
        var tmp : [lower..upper] int;
        for i in lower..upper{
            tmp[i] = x[i];
        }
        
        randStreamInt.shuffle(tmp);
        for i in lower..upper{
            x[i] = tmp[i];
        }
}


proc shuffle(ref x: [] int){
    writeln("SHUFFLING");
    for i in 0..(N/size){
        writeln("Iteration ", i);
        const start = i * size;
        const end = min( (i + 1) * size - 1, N - 1);
        writeln("start ", start);
        writeln("end ", end);
        shuffleRange(x, start, end);
                
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
    log(i,j,n);
    
    
    var randStream = new randomStream(bool);
    for count in 1..100{
        if randStream.next() {
            //writeln("True");
            if (i==j){
                break;
            }
        } else {
            //writeln("False");
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


proc mergeSort(ref x: [] int){
    for i in 0..(N/size){
        const start = i * size;
        const size1 = min( size , N - start);
        const size2 = min( 2 * size, N - start - size1);
        writeln("start: ", start);
        writeln("size1: ", size1);
        writeln("size2: ", size2);
        merge(x, start, size1, size2);
        
    }
}


writeln("N: ", N);
writeln("K: ", K);
writeln("size: ", size);
writeln("rands: ", rands);
writeln("x: ", x);

shuffle(x);
mergeSort(x);
writeln("x: ", x);

