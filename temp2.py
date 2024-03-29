import arkouda.numpy as aknp
import inspect


def write_stub(module, filename):
    with open(filename, 'w') as f:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                f.write('\n')
                f.write(f'class {name}:\n')

                for func_name, func in inspect.getmembers(obj):
                    if not func_name.startswith('__'):
                        try:
                            f.write(f'    def {func_name} {inspect.signature(func)}:\n')
                        except:
                            f.write(f'    def {func_name} (self, *args, **kwargs):\n')
                        f.write(f"      '''{func.__doc__}")
                        f.write("      '''")
                        f.write('\n    ...\n')
                   
                   
write_stub(aknp, '_numpy.pyi')
