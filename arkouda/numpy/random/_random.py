from arkouda.numpy.random._generator import default_rng as ak_default_rng


def rand(*arg):
    """
    Random values in a given shape.
    """
    rng = ak_default_rng()
    return rng.uniform(size=arg)
