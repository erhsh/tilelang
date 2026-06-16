"""Diff utilities for lower trace."""

from __future__ import annotations

import difflib


_ANSI_RESET = "\033[0m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_CYAN = "\033[36m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _inline_diff(line_before: str, line_after: str) -> tuple[str, str, bool]:
    """Compute character-level inline diff between two lines.

    Returns (left_html, right_html, is_ws_only) with highlighted changes.
    """
    sm = difflib.SequenceMatcher(None, line_before, line_after)
    left_parts = []
    right_parts = []
    is_ws_only = True

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            eq = _esc(line_before[i1:i2])
            left_parts.append(eq)
            right_parts.append(eq)
        else:
            left_chunk = line_before[i1:i2] if i2 > i1 else ""
            right_chunk = line_after[j1:j2] if j2 > j1 else ""
            if left_chunk.strip() != "" or right_chunk.strip() != "":
                is_ws_only = False

            if tag == "replace":
                left_parts.append(f'<span class="del-word">{_esc(left_chunk)}</span>')
                right_parts.append(f'<span class="add-word">{_esc(right_chunk)}</span>')
            elif tag == "delete":
                left_parts.append(f'<span class="del-word">{_esc(left_chunk)}</span>')
            elif tag == "insert":
                right_parts.append(f'<span class="add-word">{_esc(right_chunk)}</span>')

    return "".join(left_parts), "".join(right_parts), is_ws_only


def _merge_whitespace_diffs(opcodes: list, before_lines: list, after_lines: list) -> list:
    """Post-process opcodes to merge adjacent delete+insert into replace when whitespace-only."""
    result = []
    i = 0
    while i < len(opcodes):
        tag, i1, i2, j1, j2 = opcodes[i]

        if tag == "delete" and i + 1 < len(opcodes) and opcodes[i + 1][0] == "insert":
            _, _, _, j1_next, j2_next = opcodes[i + 1]
            del_lines = list(range(i1, i2))
            ins_lines = list(range(j1_next, j2_next))

            matched_del = set()
            matched_ins = set()
            pairs = []

            for di, d_idx in enumerate(del_lines):
                for ii, i_idx in enumerate(ins_lines):
                    if ii in matched_ins:
                        continue
                    if before_lines[d_idx].strip() == after_lines[i_idx].strip():
                        pairs.append((d_idx, i_idx))
                        matched_del.add(di)
                        matched_ins.add(ii)
                        break

            if pairs:
                pairs.sort()
                prev_d = i1
                prev_i = j1_next
                for d_idx, i_idx in pairs:
                    while prev_d < d_idx:
                        result.append(("delete", prev_d, prev_d + 1, prev_i, prev_i))
                        prev_d += 1
                    while prev_i < i_idx:
                        result.append(("insert", d_idx, d_idx, prev_i, prev_i + 1))
                        prev_i += 1
                    result.append(("replace", d_idx, d_idx + 1, i_idx, i_idx + 1))
                    prev_d = d_idx + 1
                    prev_i = i_idx + 1
                for d_idx in range(prev_d, i2):
                    result.append(("delete", d_idx, d_idx + 1, prev_i, prev_i))
                for i_idx in range(prev_i, j2_next):
                    result.append(("insert", i2, i2, i_idx, i_idx + 1))

                i += 2
                continue

        elif tag == "insert" and i + 1 < len(opcodes) and opcodes[i + 1][0] == "delete":
            _, i1_next, i2_next, _, _ = opcodes[i + 1]
            ins_lines = list(range(j1, j2))
            del_lines = list(range(i1_next, i2_next))

            matched_ins = set()
            matched_del = set()
            pairs = []

            for ii, i_idx in enumerate(ins_lines):
                for di, d_idx in enumerate(del_lines):
                    if di in matched_del:
                        continue
                    if after_lines[i_idx].strip() == before_lines[d_idx].strip():
                        pairs.append((d_idx, i_idx))
                        matched_ins.add(ii)
                        matched_del.add(di)
                        break

            if pairs:
                pairs.sort()
                prev_d = i1_next
                prev_i = j1
                for d_idx, i_idx in pairs:
                    while prev_i < i_idx:
                        result.append(("insert", prev_d, prev_d, prev_i, prev_i + 1))
                        prev_i += 1
                    while prev_d < d_idx:
                        result.append(("delete", prev_d, prev_d + 1, prev_i, prev_i))
                        prev_d += 1
                    result.append(("replace", d_idx, d_idx + 1, i_idx, i_idx + 1))
                    prev_d = d_idx + 1
                    prev_i = i_idx + 1
                for i_idx in range(prev_i, j2):
                    result.append(("insert", prev_d, prev_d, i_idx, i_idx + 1))
                for d_idx in range(prev_d, i2_next):
                    result.append(("delete", d_idx, d_idx + 1, j2, j2))

                i += 2
                continue

        result.append((tag, i1, i2, j1, j2))
        i += 1

    return result


def _make_diff_html(before_text: str, after_text: str, context: int = 3) -> str:
    """Generate a GitHub-style side-by-side diff HTML table."""
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()

    sm = difflib.SequenceMatcher(None, before_lines, after_lines)
    opcodes = sm.get_opcodes()

    opcodes = _merge_whitespace_diffs(opcodes, before_lines, after_lines)

    if all(tag == "equal" for tag, *_ in opcodes):
        return '<p class="noop-msg">No differences.</p>'

    before_collapse = [True] * len(before_lines)
    after_collapse = [True] * len(after_lines)
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            for i in range(i1, i2):
                before_collapse[i] = False
            for j in range(j1, j2):
                after_collapse[j] = False
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(min(context, i2 - i1)):
                before_collapse[i1 + k] = False
                after_collapse[j1 + k] = False
            for k in range(min(context, i2 - i1)):
                before_collapse[i2 - 1 - k] = False
                after_collapse[j2 - 1 - k] = False

    rows = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                collapsed = before_collapse[i]
                hidden_attr = ' class="row-hidden" data-collapse="1"' if collapsed else ""
                ln_l = f'<td class="ln ln-eq" data-side="l" data-idx="{i}">{i + 1}</td>'
                ln_r = f'<td class="ln ln-eq" data-side="r" data-idx="{j}">{j + 1}</td>'
                rows.append(
                    f"<tr{hidden_attr}>{ln_l}"
                    f'<td class="sg"></td><td class="eq">{_esc(before_lines[i])}</td>'
                    f"{ln_r}"
                    f'<td class="sg"></td><td class="eq">{_esc(after_lines[j])}</td></tr>'
                )
        elif tag == "replace":
            left_indices = list(range(i1, i2))
            right_indices = list(range(j1, j2))

            pairs = []
            used_left = set()
            used_right = set()

            for li in left_indices:
                for ri in right_indices:
                    if ri in used_right:
                        continue
                    if before_lines[li].strip() == after_lines[ri].strip():
                        pairs.append((li, ri))
                        used_left.add(li)
                        used_right.add(ri)
                        break

            remaining_left = [i for i in left_indices if i not in used_left]
            remaining_right = [j for j in right_indices if j not in used_right]

            all_rows = []

            for li, ri in pairs:
                all_rows.append((li, ri, True))

            for k in range(max(len(remaining_left), len(remaining_right))):
                if k < len(remaining_left) and k < len(remaining_right):
                    all_rows.append((remaining_left[k], remaining_right[k], False))
                elif k < len(remaining_left):
                    all_rows.append((remaining_left[k], None, False))
                else:
                    all_rows.append((None, remaining_right[k], False))

            def sort_key(row):
                li, ri, _ = row
                if li is not None and ri is not None:
                    return min(li * 1000, ri * 1000)
                elif li is not None:
                    return li * 1000
                else:
                    return ri * 1000

            all_rows.sort(key=sort_key)

            for li, ri, is_matched in all_rows:
                if li is not None and ri is not None:
                    left_html, right_html, is_ws_only = _inline_diff(before_lines[li], after_lines[ri])
                    if is_ws_only and is_matched:
                        ln_l = f'<td class="ln ln-ws" data-side="l" data-idx="{li}">{li + 1}</td>'
                        ln_r = f'<td class="ln ln-ws" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                        rows.append(
                            f"<tr>{ln_l}"
                            f'<td class="sg sg-ws">~</td><td class="ws">{left_html}</td>'
                            f"{ln_r}"
                            f'<td class="sg sg-ws">~</td><td class="ws">{right_html}</td></tr>'
                        )
                    else:
                        ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{li}">{li + 1}</td>'
                        ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                        rows.append(
                            f"<tr>{ln_l}"
                            f'<td class="sg sg-del">\u2212</td><td class="del">{left_html}</td>'
                            f"{ln_r}"
                            f'<td class="sg sg-add">+</td><td class="add">{right_html}</td></tr>'
                        )
                elif li is not None:
                    ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{li}">{li + 1}</td>'
                    rows.append(
                        f"<tr>{ln_l}"
                        f'<td class="sg sg-del">\u2212</td><td class="del">{_esc(before_lines[li])}</td>'
                        f'<td class="ln"></td><td class="sg"></td><td></td></tr>'
                    )
                else:
                    ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                    rows.append(
                        f'<tr><td class="ln"></td><td class="sg"></td><td></td>'
                        f"{ln_r}"
                        f'<td class="sg sg-add">+</td><td class="add">{_esc(after_lines[ri])}</td></tr>'
                    )
        elif tag == "delete":
            for i in range(i1, i2):
                ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{i}">{i + 1}</td>'
                rows.append(
                    f"<tr>{ln_l}"
                    f'<td class="sg sg-del">\u2212</td><td class="del">{_esc(before_lines[i])}</td>'
                    f'<td class="ln"></td><td class="sg"></td><td></td></tr>'
                )
        elif tag == "insert":
            for j in range(j1, j2):
                ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{j}">{j + 1}</td>'
                rows.append(
                    f'<tr><td class="ln"></td><td class="sg"></td><td></td>'
                    f"{ln_r}"
                    f'<td class="sg sg-add">+</td><td class="add">{_esc(after_lines[j])}</td></tr>'
                )

    return (
        '<div class="diff-table-wrap">'
        "<table><colgroup>"
        '<col style="width:50px"><col style="width:20px"><col>'
        '<col style="width:50px"><col style="width:20px"><col>'
        "</colgroup>" + "\n".join(rows) + "</table></div>"
    )


def unified_diff(
    before_text: str,
    after_text: str,
    before_label: str = "before",
    after_label: str = "after",
    context: int = 3,
    color: bool = True,
) -> str:
    """Generate a unified diff string, optionally with terminal ANSI colors."""
    before_lines = before_text.splitlines(keepends=True)
    after_lines = after_text.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=before_label,
        tofile=after_label,
        n=context,
    ))

    if not diff:
        return ""

    if not color:
        return "".join(diff)

    colored = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            colored.append(f"{_ANSI_BOLD}{line}{_ANSI_RESET}")
        elif line.startswith("@@"):
            colored.append(f"{_ANSI_CYAN}{line}{_ANSI_RESET}")
        elif line.startswith("-"):
            colored.append(f"{_ANSI_RED}{line}{_ANSI_RESET}")
        elif line.startswith("+"):
            colored.append(f"{_ANSI_GREEN}{line}{_ANSI_RESET}")
        else:
            colored.append(line)

    return "".join(colored)


def print_diff(
    before_text: str,
    after_text: str,
    before_label: str = "before",
    after_label: str = "after",
    context: int = 3,
    color: bool = True,
) -> bool:
    """Print a unified diff to stdout. Returns True if there were differences."""
    result = unified_diff(before_text, after_text, before_label, after_label, context, color)
    if result:
        print(result, end="")
        return True
    return False


def _count_changes(diff_lines: list[str]) -> tuple[int, int]:
    """Count insertions and deletions from a unified diff."""
    insertions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return insertions, deletions
