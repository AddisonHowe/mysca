"""Mapping class for AA dictionaries

"""

class SymMap:

    def __init__(self, aa_syms: str, gapsym: str, exclude_syms: str=""):
        self.aa_list = list(aa_syms)
        self.gapsym = gapsym
        self.sym_list = self.aa_list + [gapsym]
        self.exclude_syms = list(exclude_syms)
        self.sym2int = {sym: i for i, sym in enumerate(self.sym_list)}
        self.aa2int = {
            k: v for k, v in self.sym2int.items() if k in self.aa_list
        }
        self.gapint = self.sym2int[self.gapsym]

    def __getitem__(self, key):
        return self.sym2int[key]
    
    def __len__(self):
        return len(self.sym2int)


DEFAULT_MAP = SymMap(
    "ACDEFGHIKLMNPQRSTVWY", "-", []
)

