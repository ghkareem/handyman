import h5py

def print_h5_tree(node,
            *,
            max_depth=None,
            branch_head=None,
            branch_tail=0,
            skip_singleton_wrappers=True,
            _pre='',
            _depth=0,
            _apply_trim_here=False):
    """
    Pretty-print an HDF5 tree with smart trimming:
    - Skip top-level wrapper groups that have exactly one child until the first
      level that actually branches (>=2 keys). Apply head/tail trimming there.
    - Optionally limit depth with ellipses.

    Parameters
    ----------
    node : h5py.File or h5py.Group
    max_depth : int or None
        If set, limit recursion to this many levels (counting from root=0).
    branch_head : int or None
        Number of items to show at the first branching level (head). If None, show all.
    branch_tail : int
        Number of items from the end to also show at that level.
    skip_singleton_wrappers : bool
        If True, walk past chains of groups with exactly one child before applying trimming.
    """

    def _is_group(x):
        return isinstance(x, (h5py.Group, h5py.File))

    def _print_dataset_line(pre, branch, key, dset):
        # Dataset: try len(); else shape; else scalar
        try:
            n = len(dset)
            print(pre + branch + f"{key} ({n})")
        except TypeError:
            shape = getattr(dset, "shape", None)
            if not shape or shape == ():
                print(pre + branch + f"{key} (scalar)")
            else:
                print(pre + branch + f"{key} shape={shape}")

    # Depth cut
    if (max_depth is not None) and (_depth >= max_depth):
        # If we’re at/beyond limit, summarize children (if this is a group)
        if _is_group(node):
            try:
                n_hidden = len(node.keys())
            except Exception:
                n_hidden = 0
            print(_pre + "└── …" + (f" ({n_hidden} items hidden)" if n_hidden else ""))
        return

    # If we’re at the very top call, discover the first branching level
    if _depth == 0 and skip_singleton_wrappers:
        cur = node
        pre = _pre
        chain_keys = []

        # Walk while (group and exactly one key)
        while _is_group(cur):
            keys = list(cur.keys())
            if len(keys) != 1:
                break
            k = keys[0]
            child = cur[k]

            # Print this singleton link
            # Decide branch art: since it's a single child, it's drawn as '└──'
            print(pre + "└── " + k)

            # Prepare for next depth
            next_pre = pre + "    "
            chain_keys.append(k)
            cur = child
            pre = next_pre

            # Depth check while descending through wrappers
            if (max_depth is not None) and (len(chain_keys) >= max_depth):
                # We hit depth cap while descending wrappers — summarize below
                if _is_group(cur):
                    try:
                        n_hidden = len(cur.keys())
                    except Exception:
                        n_hidden = 0
                    print(pre + "└── …" + (f" ({n_hidden} items hidden)" if n_hidden else ""))
                return

        # Now 'cur' is either a dataset or a group with 0 or >=2 keys.
        if not _is_group(cur):
            # Leaf dataset at end of wrappers
            _print_dataset_line(pre, "└── ", cur.name.split('/')[-1] or "<dataset>", cur)
            return

        # We are at the first branching level -> apply trimming here.
        # Print children of `cur` with head/tail trimming, then recurse normally.
        _print_group(cur, pre, depth=len(chain_keys), apply_trim_here=True)
        return

    # If we didn’t return above, print this node normally
    if _is_group(node):
        _print_group(node, _pre, _depth, apply_trim_here=_apply_trim_here)
    else:
        _print_dataset_line(_pre, "└── ", node.name.split('/')[-1] or "<dataset>", node)


def _print_group(group, pre, depth, apply_trim_here=False,
                 *, max_depth=None, branch_head=None, branch_tail=0,
                 skip_singleton_wrappers=True):
    """
    Print a group's immediate children, with optional head/tail trimming
    (only when apply_trim_here=True). Then recurse into children.
    """
    # Collect keys deterministically
    keys = list(group.keys())

    # Decide which keys to show
    show_keys = keys
    use_ellipsis = False
    mid_hidden = 0

    if apply_trim_here and (branch_head is not None or branch_tail):
        head = branch_head if (branch_head is not None) else len(keys)
        tail = branch_tail if branch_tail else 0
        head = max(0, min(head, len(keys)))
        tail = max(0, min(tail, len(keys) - head))
        if head + tail < len(keys):
            show_keys = keys[:head] + (keys[-tail:] if tail > 0 else [])
            mid_hidden = len(keys) - len(show_keys)
            use_ellipsis = mid_hidden > 0

    # We’ll print children; if an ellipsis is needed, place it between head and tail
    total_lines = len(show_keys) + (1 if use_ellipsis else 0)
    printed_lines = 0

    # Helper to decide tree branches and recurse
    def _emit_child(idx, k, is_last_override=None):
        nonlocal printed_lines
        child = group[k]
        is_last = (printed_lines == total_lines - 1) if is_last_override is None else is_last_override
        branch = "└── " if is_last else "├── "
        next_pre = pre + ("    " if is_last else "│   ")

        if isinstance(child, h5py.Group):
            print(pre + branch + k)
            # Recurse (no trimming below the branch level)
            h5_tree(child,
                    max_depth=max_depth,
                    branch_head=branch_head,
                    branch_tail=branch_tail,
                    skip_singleton_wrappers=skip_singleton_wrappers,
                    _pre=next_pre,
                    _depth=depth+1,
                    _apply_trim_here=False)
        else:
            # Dataset
            try:
                n = len(child)
                print(pre + branch + f"{k} ({n})")
            except TypeError:
                shape = getattr(child, "shape", None)
                if not shape or shape == ():
                    print(pre + branch + f"{k} (scalar)")
                else:
                    print(pre + branch + f"{k} shape={shape}")
        printed_lines += 1

    # Print head block
    head_len = len(show_keys) if not use_ellipsis else (len(show_keys) - (branch_tail if apply_trim_here else 0))
    for i, k in enumerate(show_keys[:head_len]):
        # If an ellipsis will be printed next and nothing else after it, this could affect lastness.
        _emit_child(i, k)

    # Ellipsis line if applicable
    if use_ellipsis:
        # Ellipsis is last if there is no tail
        is_last = (branch_tail == 0)
        branch = "└── " if is_last else "├── "
        print(pre + branch + f"… ({mid_hidden} items hidden)")
        printed_lines += 1

    # Tail block
    if use_ellipsis and branch_tail:
        tail_keys = show_keys[-branch_tail:]
        for j, k in enumerate(tail_keys):
            # Last line iff this is the last tail entry
            is_last_override = (j == len(tail_keys) - 1)
            _emit_child(head_len + j, k, is_last_override=is_last_override)
