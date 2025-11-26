import h5py

def h5_tree(node, *, branch_head=None, branch_tail=0, max_depth=None, skip_singletons=True):
    """
    Print an HDF5 tree.
    - Skip chains of single-child wrapper groups from the root.
    - Apply head/tail trimming at the first level that actually branches (>=2 keys).
    - Optionally cap expansion depth (with ellipsis).
    """
    def _is_group(x):
        return isinstance(x, (h5py.Group, h5py.File))

    def _print_dataset(pre, branch, key, dset):
        # Try len(); else shape; else scalar
        try:
            n = len(dset)
            print(pre + branch + f"{key} ({n})")
        except TypeError:
            shape = getattr(dset, "shape", None)
            if not shape or shape == ():
                print(pre + branch + f"{key} (scalar)")
            else:
                print(pre + branch + f"{key} shape={shape}")

    def _subtree(obj, pre, depth):
        # Depth cap
        if max_depth is not None and depth >= max_depth:
            if _is_group(obj):
                try:
                    n_hidden = len(obj.keys())
                except Exception:
                    n_hidden = 0
                print(pre + f"└── … ({n_hidden} items hidden)" if n_hidden else pre + "└── …")
            return

        if not _is_group(obj):
            _print_dataset(pre, "└── ", obj.name.split("/")[-1] or "<dataset>", obj)
            return

        keys = list(obj.keys())
        n = len(keys)
        for i, k in enumerate(keys):
            child = obj[k]
            is_last = (i == n - 1)
            branch = "└── " if is_last else "├── "
            next_pre = pre + ("    " if is_last else "│   ")
            if _is_group(child):
                print(pre + branch + k)
                _subtree(child, next_pre, depth + 1)
            else:
                _print_dataset(pre, branch, k, child)

    # 1) Walk through top-level singleton wrappers (if requested)
    cur = node
    pre = ""
    depth = 0
    if skip_singletons:
        while _is_group(cur):
            keys = list(cur.keys())
            if len(keys) != 1:
                break
            k = keys[0]
            # Draw the chain node
            print(pre + "└── " + k)
            cur = cur[k]
            pre = pre + "    "
            depth += 1
            if max_depth is not None and depth >= max_depth:
                # reached depth cap while walking wrappers
                if _is_group(cur):
                    try:
                        n_hidden = len(cur.keys())
                    except Exception:
                        n_hidden = 0
                    print(pre + (f"└── … ({n_hidden} items hidden)" if n_hidden else "└── …"))
                return

    # 2) We are at the first branching level (or a dataset/empty group)
    if not _is_group(cur):
        # Leaf after wrappers
        _print_dataset(pre, "└── ", cur.name.split("/")[-1] or "<dataset>", cur)
        return

    keys = list(cur.keys())
    n = len(keys)

    # If it still doesn't branch (0 or 1), just print it normally
    if n <= 1 or (branch_head is None and branch_tail == 0):
        # no trimming — print full subtree from here
        _subtree(cur, pre, depth)
        return

    # 3) Apply head/tail trimming at this branching level
    head = branch_head if branch_head is not None else n
    tail = branch_tail if branch_tail else 0
    head = max(0, min(head, n))
    tail = max(0, min(tail, n - head))

    if head + tail >= n:
        show = keys
        mid_hidden = 0
        use_ellipsis = False
    else:
        show = keys[:head] + (keys[-tail:] if tail > 0 else [])
        mid_hidden = n - len(show)
        use_ellipsis = mid_hidden > 0

    # Total printed lines on this level = children shown + optional ellipsis
    total_lines = len(show) + (1 if use_ellipsis else 0)
    printed = 0

    # Helper to emit one child and recurse below (respecting max_depth)
    def emit_child(k, is_last_line):
        nonlocal printed
        child = cur[k]
        branch = "└── " if is_last_line else "├── "
        next_pre = pre + ("    " if is_last_line else "│   ")
        if _is_group(child):
            print(pre + branch + k)
            _subtree(child, next_pre, depth + 1)
        else:
            _print_dataset(pre, branch, k, child)
        printed += 1

    # Head block
    for i, k in enumerate(show[:head]):
        # Last line only if nothing else (no ellipsis, no tail, no more head)
        is_last_line = (printed == total_lines - 1)
        emit_child(k, is_last_line)

    # Ellipsis in between head and tail
    if use_ellipsis:
        # Ellipsis is last iff no tail
        is_last_line = (branch_tail == 0)
        branch = "└── " if is_last_line else "├── "
        print(pre + branch + f"… ({mid_hidden} items hidden)")
        printed += 1

    # Tail block
    tail_keys = show[-tail:] if tail > 0 else []
    for j, k in enumerate(tail_keys):
        is_last_line = (printed == total_lines - 1)
        emit_child(k, is_last_line)
