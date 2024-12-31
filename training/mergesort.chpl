
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


proc shuffle(ref x: [] int){
    writeln("SHUFFLING");
    for i in 0..(N/size){
        writeln("Iteration ", i);
        const start = i * size;
        const end = min( (i + 1) * size - 1, N - 1);
        writeln("start ", start);
        writeln("end ", end);
        var tmp : [start..end] int;
        for i in start..end{
            tmp[i] = x[i];
        }
        
        randStreamInt.shuffle(tmp);
        for i in start..end{
            x[i] = tmp[i];
        }
                
    }
    
}


writeln("N: ", N);
writeln("K: ", K);
writeln("size: ", size);
writeln("rands: ", rands);
writeln("x: ", x);

shuffle(x);
writeln("x: ", x);

