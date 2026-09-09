import pickle
from pathlib import Path

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name in ("PosixPath", "WindowsPath"):
            return Path
        return super().find_class(module, name)