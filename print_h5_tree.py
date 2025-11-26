import h5py

def print_h5_tree(node,
            pre='',
            *,
            depth=0,
            max_depth=None,
            first_level_head=None,
            first_level_tail=0):
    """
    Pretty-print an HDF5 tree.

    Parameters
    ----------
    node : h5py.Group or h5py.File
        The starting HDF5 object.
    pre : str
        Prefix used internally for drawing branches (leave default).
    depth : int
        Current depth (leave default).
    max_depth : int or None
        If set, limits recursion to this many levels beneath the start node.
        When reached, prints an ellipsis line "… (N items hidden)" instead of expanding.
    first_level_head : int or None
        At the top level only (depth==0), show at most this many items from the start.
        If None, show all (subject to first_level_tail).
    first_level_tail : int
        At the top level only, also show this many items from the end.
        If items are hidden between head and tail, an ellipsis line is inserted.
    """
    # Decide whether we’re at/over depth limit
    if (max_depth is not None) and (depth >= max_depth):
        # At this point 'node' should be a group; if datasets are here, just note count
        try:
            n_hidden = len(node.keys()) if isinstance(node, h5py.Group) else 0
        except Exception:
            n_hidden = 0
        print(pre + f"└── … ({n_hidden} items hidden)")
        return

    # Collect keys deterministically
    if isinstance(node, (h5py.Group, h5py.File)):
        keys = list(node.keys())
    else:
        # Dataset or scalar leaf: nothing further to expand
        return

    # Apply top-level head/tail trimming only at depth==0
    show_keys = keys
    if depth == 0 and (first_level_head is not None or first_level_tail):
        head = first_level_head if (first_level_head is not None) else len(keys)
        tail = first_level_tail if first_level_tail else 0
        head = max(0, min(head, len(keys)))
        tail = max(0, min(tail, len(keys) - head))
        if head + tail >= len(keys):
            show_keys = keys  # nothing to hide
            mid_hidden = 0
            use_ellipsis = False
        else:
            show_keys = keys[:head] + keys[-tail:] if tail > 0 else keys[:head]
            mid_hidden = len(keys) - len(show_keys)
            use_ellipsis = mid_hidden > 0
    else:
        use_ellipsis = False
        mid_hidden = 0

    # Helper to print one entry (group or dataset)
    def _print_entry(is_last, k):
        val = node[k]
        branch = '└── ' if is_last else '├── '
        next_pre = pre + ('    ' if is_last else '│   ')
        if isinstance(val, h5py.Group):
            print(pre + branch + k)
            h5_tree(val, pre=next_pre,
                    depth=depth+1,
                    max_depth=max_depth,
                    first_level_head=first_level_head,
                    first_level_tail=first_level_tail)
        else:
            # Dataset: try to show length; fall back to shape or scalar
            try:
                n = len(val)
                print(pre + branch + f"{k} ({n})")
            except TypeError:
                # len() not defined; try shape/size
                try:
                    shape = getattr(val, 'shape', None)
                    if shape is None or shape == ():
                        print(pre + branch + f"{k} (scalar)")
                    else:
                        print(pre + branch + f"{k} shape={shape}")
                except Exception:
                    print(pre + branch + f"{k} (dataset)")

    # Determine where ellipsis (if any) should appear in sequence
    # For a clean look, we place it between head and tail blocks.
    printed = 0
    total_to_print = len(show_keys) + (1 if use_ellipsis else 0)
    for idx, k in enumerate(show_keys):
        # Insert ellipsis between head and tail blocks when applicable
        if use_ellipsis and depth == 0 and first_level_head is not None:
            if idx == first_level_head:
                # Ellipsis is not last unless no tail exists and no items remain
                is_last_ellipsis = (idx == len(show_keys)) and (first_level_tail == 0)
                branch = '└── ' if (total_to_print == 1) else '├── '
                # If there is a tail, the ellipsis is not the last printed entry yet
                if first_level_tail:
                    branch = '├── '
                else:
                    # No tail: ellipsis could be last if this is the final entry overall
                    branch = '└── ' if (idx == len(show_keys)) else '├── '
                print(pre + branch + f"… ({mid_hidden} items hidden)")
                printed += 1
        # Decide if current printed entry is the last one
        is_last = (printed + (len(show_keys) - idx) == 1)
        # If we still have an ellipsis to print after this, it's not the last
        if use_ellipsis and depth == 0 and first_level_tail and (idx == len(show_keys) - first_level_tail):
            # There's an ellipsis already printed; last calculation is fine
            pass
        _print_entry(is_last, k)
        printed += 1

    # Edge case: if head==0 and we haven't printed ellipsis yet, print it now
    if use_ellipsis and depth == 0 and (first_level_head == 0):
        is_last = (first_level_tail == 0)
        branch = '└── ' if is_last else '├── '
        print(pre + branch + f"… ({mid_hidden} items hidden)")
