import itertools
import json
import sys
from enum import Enum

import chapel

FUNCS = [["any", "bool"], ["all", "bool"], ["isSorted", "bool"], ["isSortedLocally", "bool"]]


def generate_function():
    ret = ""
    for func, ret_type in FUNCS:
        ret += f"""@arkouda.registerCommand
    proc {func}(ref x:[?d] ?t, axis: list(int)): [] {ret_type} throws {{
      use SliceReductionOps;

      if d.rank == 1 {{ 
        return makeDistArray([{func}(x)]);
      }}
      const (valid, axes) = validateNegativeAxes(axis, x.rank);
      if !valid {{
        throw new Error("Invalid axis value(s) '%?' in all slicing reduction".format(axis));
      }} else {{
        const outShape = reducedShape(x.shape, axes);
        var ret = makeDistArray((...outShape), bool);
        if (ret.size==1) {{
          ret[ret.domain.low] = {func}(x);
        }}else{{
          forall sliceIdx in domOffAxis(x.domain, axes) {{
            const sliceDom = domOnAxis(x.domain, sliceIdx, axes);
            ret[sliceIdx] = {func}(x, sliceDom);
          }}
        }}
        return ret;
      }}
    }}

    """

    ret = ret.replace("\t", "  ")
    ret = "    " + ret

    return ret


def main():
    print("TEST")

    infile = "/home/amandapotts/git/arkouda/src/ReductionMsg.chpl"

    x = generate_function()
    print(x)

    outfile = "/home/amandapotts/git/arkouda/test_reductions.txt"
    with open(outfile, "w") as text_file:
        text_file.write(x)


if __name__ == "__main__":
    main()
