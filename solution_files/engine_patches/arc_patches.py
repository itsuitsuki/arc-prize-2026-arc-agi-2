import importlib.abc
import importlib.util
import sys

PATCHES = {}


def chain(name, hook):
    previous = PATCHES.get(name)
    PATCHES[name] = hook if previous is None else (lambda module: (previous(module), hook(module)))


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        load = spec.loader.exec_module

        def exec_module(module):
            load(module)
            PATCHES[name](module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, Hook())
