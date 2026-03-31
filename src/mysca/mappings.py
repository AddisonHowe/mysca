"""Mapping class for AA dictionaries

"""

NONCANONICAL = "noncanonical"


class SymMap:

    def __init__(self, aa_syms: str, gapsym: str, exclude_syms=NONCANONICAL):
        self.aa_list = list(aa_syms)
        self.gapsym = gapsym
        self.sym_list = self.aa_list + [gapsym]
        self._valid_syms = set(self.sym_list)
        if exclude_syms is NONCANONICAL:
            self._exclude_noncanonical = True
            self.exclude_syms = []
        else:
            self._exclude_noncanonical = False
            self.exclude_syms = list(exclude_syms)
        self.sym2int = {sym: i for i, sym in enumerate(self.sym_list)}
        self.aa2int = {
            k: v for k, v in self.sym2int.items() if k in self.aa_list
        }
        self.gapint = self.sym2int[self.gapsym]

    def is_excluded(self, sym: str) -> bool:
        """Return True if the symbol should be excluded."""
        if self._exclude_noncanonical:
            return sym not in self._valid_syms
        return sym in self.exclude_syms

    def __getitem__(self, key):
        return self.sym2int[key]

    def __len__(self):
        return len(self.sym2int)


DEFAULT_MAP = SymMap(
    "ACDEFGHIKLMNPQRSTVWY", "-"
)

