# -*- coding: utf-8 -*-
"""
Attribute-style access & data export helpers for uproot files/trees/branches.

Author Kareem Farrag
Date 10 October 2025

Usage
-----
import uproot
from file_handler import open_as_attrs

f = uproot.open("your.root")
fh = open_as_attrs(f)

# Leaf branch (direct):
a = fh.HITS.TimeOfFlight.array(library="np")

# 1-column DataFrame from a leaf:
df_leaf = fh.DaughtersFromDecays.DaughterPDGID.to_dataframe()

# Whole TTree (all immediate children) to pandas:
df_tree = fh.DaughtersFromDecays.to_dataframe()

# Subset by safe names or original names:
df_sel = fh.DaughtersFromDecays.to_dataframe(["EventID", "DaughterPDGID"])

# Utilities:
fh.print_keys()                       # safe names at root level
fh.print_keys(original=True)          # original ROOT keys (with cycles)
fh.pretty_tree(show_original=True)    # full tree dump
"""

import re
import os
from typing import Any, Dict, Optional, Iterable, List, Tuple, Union
import numpy as np
import uproot
# Optional pandas (kept optional, build DF only if available)
try:
    import pandas as pd  # noqa: F401
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# Optional awkward (used as a robust fallback)
try:
    import awkward as ak  # noqa: F401
    _HAS_AWKWARD = True
except Exception:
    _HAS_AWKWARD = False

# Uproot type checks (v4/v5)
try:
    from uproot.behaviors.TTree import TTree
    from uproot.behaviors.TBranch import TBranch, HasBranches
except Exception:
    TTree = TBranch = HasBranches = tuple()  # type: ignore


# ----------------- helpers -----------------

_cycle_re = re.compile(r"^(?P<name>.*?);(?P<cycle>\d+)$")

def strip_cycle(key: str) -> Tuple[str, Optional[int]]:
    m = _cycle_re.match(key)
    return (m.group("name"), int(m.group("cycle"))) if m else (key, None)

def clean_branch_name(name: str) -> str:
    return strip_cycle(name)[0]

def to_safe_attr(name: str) -> str:
    safe = re.sub(r"\W|^(?=\d)", "_", name)
    return safe or "_"

def _wrap_child(child_obj: Any, clean_name: str, chosen_key: str,
                prefer_latest_cycle: bool, parent: Optional["UprootNode"]) -> Any:
    """
    IMPORTANT: Prioritize `.array` => leaf.
    Some uproot branches expose both `keys()` and `.array()`. If `.array` exists,
    we treat it as a leaf to allow direct column reads like .array()/.to_dataframe().
    """
    # 1) Anything with .array is a LEAF (covers plain TBranch and many branch behaviors)
    if hasattr(child_obj, "array"):
        return UprootLeaf(child_obj, name=clean_name, orig_key=chosen_key, parent=parent)

    # 2) TTree is a NODE
    if isinstance(child_obj, TTree):
        return UprootNode(child_obj, name=clean_name,
                          prefer_latest_cycle=prefer_latest_cycle, parent=parent)

    # 3) Split branch / directory-like → NODE
    if isinstance(child_obj, HasBranches) or hasattr(child_obj, "keys"):
        return UprootNode(child_obj, name=clean_name,
                          prefer_latest_cycle=prefer_latest_cycle, parent=parent)

    # 4) Fallback: leaf
    return UprootLeaf(child_obj, name=clean_name, orig_key=chosen_key, parent=parent)


# ----------------- wrappers -----------------

class UprootLeaf:
    """
    Leaf wrapper for branches/objects without subkeys.
    Delegates unknown attributes/methods to the wrapped uproot object.

    Data exports:
      - .to_numpy()      -> 1D NumPy array
      - .to_dataframe()  -> 1-column DataFrame (requires pandas)
    """
    def __init__(self, obj: Any, name: str, orig_key: str, parent: Optional["UprootNode"] = None):
        self.obj = obj
        self.name = name            # clean name (no cycle)
        self._orig_key = orig_key   # original key (may include cycle)
        self._parent = parent

    def __getattr__(self, attr: str):
        # Delegate to underlying uproot object (so .array(), .interpretation, etc. work)
        return getattr(self.obj, attr)

    @property
    def original_key(self) -> str:
        return self._orig_key

    def _as_numpy_1d(self, **array_kwargs) -> np.ndarray:
        """
        Return a 1D NumPy array for this branch, robustly.
        Tries: uproot leaf -> np directly; then awkward -> np fallback.
        """
        # 1) Prefer direct NumPy from uproot
        try:
            arr = self.obj.array(library="np", **array_kwargs)
            # Some environments can still yield an Awkward Array; normalize
            if _HAS_AWKWARD and "awkward" in type(arr).__module__:
                arr = ak.to_numpy(arr)  # type: ignore
            return np.asarray(arr)
        except Exception:
            # 2) Awkward fallback
            if _HAS_AWKWARD:
                arr_ak = self.obj.array(library="ak", **array_kwargs)
                return ak.to_numpy(arr_ak)  # type: ignore
            # 3) Last resort: default .array() then cast
            raw = self.obj.array(**array_kwargs)
            return np.asarray(raw)

    def to_numpy(self, **array_kwargs) -> np.ndarray:
        """Return a 1D NumPy array for this branch."""
        return self._as_numpy_1d(**array_kwargs)

    def to_dataframe(self, **array_kwargs):
        """
        Return a 1-column pandas DataFrame for this branch, built
        from the branch’s own data (never via parent.arrays()).
        """
        if not _HAS_PANDAS:
            raise RuntimeError("pandas is not installed. Use .to_numpy() instead.")
        col = self._as_numpy_1d(**array_kwargs)
        return pd.DataFrame({self.name: col})

    def pretty_tree(self, *, include_types: bool = True, show_original: bool = False):
        """
        Print a single-line summary for a leaf (for API symmetry).
        """
        parts = [self.name]
        if show_original:
            parts.append(f"(orig: {self._orig_key})")
        if include_types:
            parts.append(f"[{type(self.obj).__name__}]")
        print("- " + " ".join(parts))

    def __repr__(self):
        return f"<UprootLeaf name={self.name!r} orig_key={self._orig_key!r} obj={type(self.obj).__name__}>"

    # --- minimal dict-like surface so leaf calls don't crash ---
    def keys(self, *, original: bool = False):  # pragma: no cover - convenience
        """Leaf has no children; returns an empty iterable (for API symmetry)."""
        return []

    def print_keys(self, *, original: bool = False) -> None:  # pragma: no cover - convenience
        """Leaf has no children; print a short hint instead of raising."""
        print("(leaf) no child keys")


class UprootNode:
    """
    Attribute-access wrapper around hierarchical uproot objects (files, dirs, TTrees, split branches).

    - Attribute access: node.HITS.TimeOfFlight
    - Item access:      node['HITS;1'] or node['HITS']
    - Data export on TTrees / split branches:
        * .to_dataframe(columns=None, **arrays_kwargs)
        * .to_numpy_dict(columns=None, **arrays_kwargs)
        * .to_numpy_structured(columns=None, dtype=None, **arrays_kwargs)
    """

    def __init__(self, obj: Any, name: str = "<root>", *,
                 prefer_latest_cycle: bool = True,
                 parent: Optional["UprootNode"] = None):
        self.obj = obj
        self.name = name
        self._prefer_latest_cycle = prefer_latest_cycle
        self._parent = parent

        # Maps safe attribute -> original key (possibly with cycle)
        self._attr_to_key: Dict[str, str] = {}
        # Child objects: safe attribute -> UprootNode or UprootLeaf
        self._children: Dict[str, Any] = {}

        self._build_index()

    # ---------- core indexing ----------
    def _expose_child_attr(self, safe: str, child: Any) -> None:
        """
        Make child visible as a real attribute to support TAB completion.
        Avoid clobbering class attributes/methods.
        """
        # Only assign if not shadowing a class attribute or an existing instance attr
        if (safe not in self.__dict__) and (not hasattr(type(self), safe)):
            self.__dict__[safe] = child  # bypass __setattr__ to keep it simple

    def _build_index(self):
        if not hasattr(self.obj, "keys"):
            return

        # Group by clean name, track cycles
        grouped: Dict[str, Dict[str, int]] = {}
        for key in self.obj.keys():
            clean, cyc = strip_cycle(key)
            grouped.setdefault(clean, {})
            grouped[clean][key] = (cyc if cyc is not None else -1)

        # Resolve the representative key for each clean name
        resolved: Dict[str, str] = {}
        for clean_name, key_to_cycle in grouped.items():
            if self._prefer_latest_cycle and key_to_cycle:
                best_key = max(key_to_cycle.items(), key=lambda kv: kv[1])[0]
            else:
                best_key = sorted(key_to_cycle.keys())[0]
            resolved[clean_name] = best_key

        # Create children
        for clean_name, chosen_key in resolved.items():
            safe = to_safe_attr(clean_name)
            base, i = safe, 2
            while safe in self._attr_to_key and self._attr_to_key[safe] != chosen_key:
                safe = f"{base}__{i}"; i += 1
            self._attr_to_key[safe] = chosen_key

            try:
                child_obj = self.obj[chosen_key]
            except Exception:
                continue

            child = _wrap_child(child_obj, clean_name, chosen_key,
                                self._prefer_latest_cycle, parent=self)
            self._children[safe] = child
            self._expose_child_attr(safe, child)  # <--- make TAB-completable

    # ---------- attribute & item access ----------
    def __getattr__(self, name: str):
        if name in self._children:
            return self._children[name]
        # delegate to underlying uproot object (e.g., TTree.arrays, .show)
        if hasattr(self.obj, name):
            return getattr(self.obj, name)
        if hasattr(self.obj, "keys"):
            raise AttributeError(
                f"'{self.name}' has no attribute '{name}'. "
                f"Available: {', '.join(sorted(self._children.keys()))}"
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str):
        # Try original key directly
        try:
            child_obj = self.obj[key]
        except Exception:
            # Try by clean name (ignore cycle)
            clean, _ = strip_cycle(key)
            for orig in self._attr_to_key.values():
                if strip_cycle(orig)[0] == clean:
                    child_obj = self.obj[orig]
                    break
            else:
                raise KeyError(f"Key '{key}' not found")
        if hasattr(child_obj, "keys"):
            return UprootNode(child_obj, name=key,
                              prefer_latest_cycle=self._prefer_latest_cycle, parent=self)
        return UprootLeaf(child_obj, name=key, orig_key=key, parent=self)

    # ---------- utilities ----------
    def keys(self, *, original: bool = False) -> Iterable[str]:
        """
        List keys at this level.
        - original=False: safe attribute names (default)
        - original=True:  original ROOT keys (may include cycles)
        """
        return self._children.keys() if not original else [self._attr_to_key[s] for s in self._children]

    def items(self, *, original: bool = False):
        """
        Like dict.items() for children. Yields (name, child) pairs.
        Name is safe attr by default, or original ROOT key if original=True.
        """
        if not original:
            for s, child in self._children.items():
                yield s, child
        else:
            for s, child in self._children.items():
                yield self._attr_to_key[s], child

    def print_keys(self, *, original: bool = False) -> None:
        """Print just the keys at this level (no recursion)."""
        for k in self.keys(original=original):
            print(k)

    def walk(self, *, leaves_only: bool = False):
        """
        Generator over the subtree: yields (path, obj) where path uses SAFE names.
        Set leaves_only=True to yield only leaves (UprootLeaf).
        """
        def _rec(node, prefix):
            for safe, child in node._children.items():
                path = f"{prefix}/{safe}" if prefix else safe
                if isinstance(child, UprootNode):
                    if not leaves_only:
                        yield (path, child)
                    yield from _rec(child, path)
                else:
                    yield (path, child)
        yield from _rec(self, "")

    def pretty_tree(
        self,
        *,
        include_types: bool = True,
        show_original: bool = False,
        max_depth: Optional[int] = None,
        indent: str = "  ",
        file=None,
    ) -> None:
        """Pretty-print a tree of keys starting at this node."""
        import sys
        out = file or sys.stdout

        def _typename(o: Any) -> str:
            try:
                return type(o.obj if hasattr(o, "obj") else o).__name__
            except Exception:
                return "Unknown"

        def _line(name: str, child) -> str:
            parts = [name]
            if show_original and isinstance(child, (UprootNode, UprootLeaf)):
                for safe, orig in self._attr_to_key.items():
                    if self._children.get(safe) is child:
                        parts.append(f"(orig: {orig})")
                        break
            if include_types:
                parts.append(f"[{_typename(child)}]")
            return " ".join(parts)

        def _rec(node: "UprootNode", depth: int):
            if max_depth is not None and depth > max_depth:
                return
            for safe, child in sorted(node._children.items()):
                print(f"{indent*depth}- {_line(safe, child)}", file=out)
                if isinstance(child, UprootNode):
                    if max_depth is None or depth < max_depth:
                        _rec(child, depth + 1)

        print(f"{self.name} [{type(self.obj).__name__}]", file=out)
        _rec(self, 1)

    def memory_report(self, *, show_branches: bool = False, return_data: bool = False):
        """
        Traverse all levels under this node and report size metrics for every TTree/TNtuple.

        Parameters
        ----------
        show_branches : bool
            If True, include per-branch size breakdown for each tree.
        return_data : bool
            If True, return a list of dicts with the results instead of only printing.

        Returns
        -------
        list[dict] if return_data else None
        """
        rows: List[Dict[str, Any]] = []
        total_unc = 0
        total_zip = 0

        for path, obj in self.walk(leaves_only=False):
            if isinstance(obj, UprootNode) and _is_uproot_tree(obj):
                tree = obj.obj
                # Entries
                try:
                    entries = int(getattr(tree, "num_entries"))
                except Exception:
                    entries = int(getattr(tree, "num_entries", 0))

                # Bytes
                unc, z = _get_bytes_pair_from_uproot_obj(tree)
                total_unc += unc
                total_zip += z
                rows.append({
                    "path": path,
                    "type": "TTree",
                    "entries": entries,
                    "uncompressed_bytes": unc,
                    "compressed_bytes": z,
                    "compression_ratio": _safe_ratio(unc, z),
                })

                if show_branches:
                    for bname, bobj in _iter_branches(tree):
                        b_unc, b_zip = _get_bytes_pair_from_uproot_obj(bobj)
                        rows.append({
                            "path": _join_path(path, bname),
                            "type": "TBranch",
                            "entries": entries,
                            "uncompressed_bytes": b_unc,
                            "compressed_bytes": b_zip,
                            "compression_ratio": _safe_ratio(b_unc, b_zip),
                        })

        # Pretty print
        def _print_table():
            if not rows:
                print("(no TTrees/TNtuples found)")
                return
            # header
            print(f"{'PATH':<60} {'TYPE':<8} {'ENTRIES':>12} {'UNC':>14} {'ZIP':>14} {'RATIO':>8}")
            print("-" * 118)
            for r in rows:
                print(
                    f"{r['path']:<60} {r['type']:<8} "
                    f"{r['entries']:>12,} "
                    f"{_human_bytes(r['uncompressed_bytes']):>14} "
                    f"{_human_bytes(r['compressed_bytes']):>14} "
                    f"{r['compression_ratio']:>8.2f}"
                )
            print("-" * 118)
            ratio = _safe_ratio(total_unc, total_zip)
            print(
                f"{'TOTAL':<60} {'':<8} {'':>12} "
                f"{_human_bytes(total_unc):>14} {_human_bytes(total_zip):>14} {ratio:>8.2f}"
            )

        _print_table()
        if return_data:
            return rows

    def __dir__(self):
        """
        Help tab-completion: include class attrs, instance attrs, and child names.
        """
        std = set(super().__dir__())
        std.update(self.__dict__.keys())
        std.update(self._children.keys())
        return sorted(std)

    def __repr__(self):
        return f"<UprootNode name={self.name!r} n_children={len(self._children)}>"

    # ---------- predicates ----------
    @property
    def is_tree(self) -> bool:
        return isinstance(self.obj, TTree)

    @property
    def is_split_branch(self) -> bool:
        return isinstance(self.obj, HasBranches)

    # ---------- column selection ----------
    def _branch_keys_for_selection(self, columns: Optional[Iterable[str]] = None) -> List[str]:
        """
        Build a list of branch names for uproot.arrays.
        Accepts safe names or original names; always strips cycles for safety.
        """
        if columns is None:
            origs = [self._attr_to_key[s] for s in self._children.keys()]
        else:
            origs = [self._attr_to_key[c] if c in self._attr_to_key else c for c in columns]
        return [clean_branch_name(o) for o in origs]

    # ---------- data export ----------
    def to_dataframe(self, columns: Optional[Iterable[str]] = None, **arrays_kwargs):
        if not _HAS_PANDAS:
            raise RuntimeError("pandas is not installed. Use to_numpy_dict/structured instead.")
        if not hasattr(self.obj, "arrays"):
            raise TypeError(f"Object '{self.name}' does not support bulk arrays().")
        cols = self._branch_keys_for_selection(columns)
        return self.obj.arrays(cols, library="pd", **arrays_kwargs)

    def to_numpy_dict(self, columns: Optional[Iterable[str]] = None, **arrays_kwargs) -> Dict[str, np.ndarray]:
        if not hasattr(self.obj, "arrays"):
            raise TypeError(f"Object '{self.name}' does not support bulk arrays().")
        cols = self._branch_keys_for_selection(columns)
        return self.obj.arrays(cols, library="np", **arrays_kwargs)

    def to_numpy_structured(self,
                            columns: Optional[Iterable[str]] = None,
                            dtype: Optional[List[Tuple[str, Any]]] = None,
                            align: bool = True,
                            **arrays_kwargs) -> np.ndarray:
        data = self.to_numpy_dict(columns, **arrays_kwargs)  # dict name->array
        names = list(data.keys())
        arrays = [data[n] for n in names]

        if dtype is None:
            dtype = []
            for n, a in zip(names, arrays):
                # map original name to safe field name if possible
                field = None
                for s, orig in self._attr_to_key.items():
                    if clean_branch_name(orig) == clean_branch_name(n):
                        field = s; break
                field = field or to_safe_attr(clean_branch_name(n))
                dtype.append((field, np.asarray(a).dtype))

        rec = np.empty(len(arrays[0]) if arrays else 0, dtype=np.dtype(dtype, align=align))
        for (field, _), a in zip(dtype, arrays):
            rec[field] = np.asarray(a).astype(rec.dtype[field], copy=False)
        return rec


# ---------- factory ----------

def open_as_attrs(uproot_obj: Any, *, prefer_latest_cycle: bool = True) -> Union["UprootNode", "UprootLeaf"]:
    """
    Wrap an uproot object (file, directory, TTree, branch) for attribute access and helpers.
    """
    # Accept a file path directly (common ergonomic use-case)
    try:
        if isinstance(uproot_obj, (str, os.PathLike)):
            uproot_obj = uproot.open(uproot_obj)
    except Exception as e:  # pragma: no cover - runtime dependent
        raise RuntimeError(f"Failed to open ROOT file: {uproot_obj!r}: {e}") from e

    if hasattr(uproot_obj, "keys"):
        return UprootNode(uproot_obj, name="<root>", prefer_latest_cycle=prefer_latest_cycle, parent=None)
    return UprootLeaf(uproot_obj, name="<root>", orig_key="<root>", parent=None)


# ------- memory report helpers -------

def _human_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or u == "PB":
            return f"{n:,.2f} {u}"
        n /= 1024.0


def _get_bytes_pair_from_uproot_obj(obj) -> Tuple[int, int]:
    """
    Return (uncompressed_bytes, compressed_bytes) for a TTree/TBranch,
    working across uproot versions. Falls back to ROOT member names if needed.
    """
    # Common in uproot v4/v5 (behaviors.TTree / behaviors.TBranch):
    for unc_attr, zip_attr in (("num_bytes", "compressed_bytes"),
                               ("nbytes", "compressed_bytes")):
        unc = getattr(obj, unc_attr, None)
        z   = getattr(obj, zip_attr, None)
        if isinstance(unc, (int, float)) and isinstance(z, (int, float)):
            return int(unc), int(z)

    # Fallback to ROOT member names carried through by uproot:
    try:
        # Some uproot objects expose ROOT members via .member(...)
        tot = obj.member("fTotBytes")
        zipb = obj.member("fZipBytes")
        if isinstance(tot, (int, float)) and isinstance(zipb, (int, float)):
            return int(tot), int(zipb)
    except Exception:
        pass

    # Last resort: zeros (unknown)
    return 0, 0


def _is_uproot_tree(node: "UprootNode") -> bool:
    # Your UprootNode already has .is_tree, keep a defensive fallback:
    try:
        if node.is_tree:
            return True
    except Exception:
        pass
    # If it quacks like a TTree (has .num_entries), treat as tree
    return hasattr(getattr(node, "obj", None), "num_entries")


def _iter_branches(tree_obj):
    """Yield (name, branch_obj) pairs for all immediate branches in a TTree."""
    try:
        for k in tree_obj.keys():
            yield k, tree_obj[k]
    except Exception:
        # Some trees expose .branches
        for br in getattr(tree_obj, "branches", []):
            try:
                yield getattr(br, "name", br.__class__.__name__), br
            except Exception:
                continue


def _safe_ratio(u: int, z: int) -> float:
    return (float(u) / float(z)) if z else float("inf")


def _join_path(parent: str, child: str) -> str:
    return f"{parent}/{child}" if parent else child


# (memory_report is now a proper method on UprootNode; no monkey-patch needed.)
