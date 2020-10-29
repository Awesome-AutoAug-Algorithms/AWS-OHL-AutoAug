from typing import Callable


class Registry(dict):
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
    
    @staticmethod
    def _register_generic(module_dict, clz_or_func: Callable, alias=None):
        if alias is None:
            alias = clz_or_func.__name__
        assert alias not in module_dict
        module_dict[alias] = clz_or_func
    
    def register(self, arg):
        if isinstance(arg, Callable):
            clz_or_func: Callable = arg
            Registry._register_generic(self, clz_or_func)
            return clz_or_func
        else:
            def register_fn(clz_or_func: Callable):
                Registry._register_generic(self, clz_or_func)
            
            return register_fn
