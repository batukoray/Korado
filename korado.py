"""Korado — Koray's Operations Research App for Decision Optimization. """

from __future__ import annotations

import argparse
import curses
import math
import re
import sys
from dataclasses import dataclass, field
import shutil
import itertools

try:
    import pulp
except ModuleNotFoundError:
    pulp = None

EPSILON = 1e-7
EXAMPLE_MODEL_LINES = [
    'MIN 50 X1 + 100 X2',
    'ST',
    '7 X1 + 2 X2 >= 28',
    '2 X1 + 12 X2 >= 24',
    'END',
]

# ── Regex building blocks ───────────────────────────────────────

# LINDO variable: alpha start, then alpha/digit/underscore/dot
_VAR = r'[A-Za-z][A-Za-z0-9_.]*'
# Number: integer, decimal, or scientific notation
_NUM = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

# Token: sign + (coeff var | bare var | bare number)
_TOKEN_RE = re.compile(
    rf'([+-]?)\s*'
    rf'(?:'
    rf'({_NUM})\s*({_VAR})'   # coefficient + variable
    rf'|({_VAR})'             # bare variable (coeff = 1)
    rf'|({_NUM})'             # bare number
    rf')'
)
_RELATION_RE = re.compile(r'(<=|>=|=|<|>)')
# Constraint label: NAME) body  **or**  NAME: body  (both accepted)
_LABEL_RE = re.compile(rf'^\s*({_VAR})\s*[):]\s*(.+)$')

# Objective keywords → canonical sense
_OBJ_KW = {
    'MAXIMIZE': 'MAX', 'MAXIMISE': 'MAX', 'MAX': 'MAX',
    'MINIMIZE': 'MIN', 'MINIMISE': 'MIN', 'MIN': 'MIN',
}


# ── Data classes ────────────────────────────────────────────────

class ModelParseError(ValueError):
    pass


@dataclass
class ConstraintSpec:
    name: str
    lhs_terms: dict[str, float]
    operator: str
    rhs: float
    source: str


@dataclass
class ModelSpec:
    sense: str
    objective_terms: dict[str, float]
    constraints: list[ConstraintSpec]
    variables: list[str]
    free_vars: set[str] = field(default_factory=set)
    integer_vars: set[str] = field(default_factory=set)
    binary_vars: set[str] = field(default_factory=set)
    lower_bounds: dict[str, float] = field(default_factory=dict)
    upper_bounds: dict[str, float] = field(default_factory=dict)
    title: str | None = None


@dataclass
class ConstraintResult:
    name: str
    slack: float
    binding: bool
    source: str


@dataclass
class SolveResult:
    status: str
    objective_value: float | None
    variable_values: dict[str, float | None]
    constraints: list[ConstraintResult]
    message: str | None = None


# ── Small helpers ───────────────────────────────────────────────

def _natural_key(text: str) -> list[object]:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text.upper())]


def _strip_comment(line: str) -> str:
    """Remove a LINDO ``!`` comment (everything from ``!`` to EOL)."""
    idx = line.find("!")
    return line[:idx] if idx != -1 else line


def _norm_kw(line: str) -> str:
    """Normalise a stripped line to a canonical section keyword."""
    n = ' '.join(line.strip().upper().replace('.', '').split())
    if n in ('ST', 'SUBJECT TO', 'SUCH THAT'):
        return 'ST'
    return n


def _detect_obj(line: str) -> tuple[str, str] | None:
    """Return (sense, expression_text) if *line* begins with an objective keyword."""
    upper = line.lstrip().upper()
    # Try longest keywords first so MAXIMIZE beats MAX, etc.
    for kw in sorted(_OBJ_KW, key=len, reverse=True):
        if upper.startswith(kw):
            rest = line.lstrip()[len(kw):]
            if rest and not rest[0].isspace():
                continue
            return _OBJ_KW[kw], rest.strip()
    return None


def _near_zero(v: float) -> float:
    return 0.0 if abs(v) <= EPSILON else v


def _fmt(v: float | None, w: int = 14) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'N/A':>{w}}"
    return f'{_near_zero(v):>{w}.6f}'


# ── Expression parser ───────────────────────────────────────────

def _parse_expr(text: str) -> tuple[dict[str, float], float]:
    """Parse a linear expression → ({var: coeff}, constant)."""
    text = text.strip()
    if not text:
        raise ModelParseError('Expected a linear expression.')

    pos, terms, const = 0, {}, 0.0

    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break

        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ModelParseError(f"Cannot parse near '{text[pos:pos+20]}'.")

        sign_s, c_num, c_var, bare_var, bare_num = m.groups()
        sign = -1.0 if sign_s == '-' else 1.0

        if c_num is not None and c_var is not None:
            v = c_var.upper()
            terms[v] = terms.get(v, 0.0) + sign * float(c_num)
        elif bare_var is not None:
            v = bare_var.upper()
            terms[v] = terms.get(v, 0.0) + sign
        elif bare_num is not None:
            const += sign * float(bare_num)
        else:
            raise ModelParseError(f"Cannot parse near '{text[pos:pos+20]}'.")

        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] not in '+-':
            raise ModelParseError(f"Expected '+' or '-' near '{text[pos:pos+20]}'.")

    return terms, const


def _parse_obj_expr(text: str) -> dict[str, float]:
    terms, _ = _parse_expr(text)
    out = {n: c for n, c in terms.items() if abs(c) > EPSILON}
    if not out:
        raise ModelParseError('Objective has no variable terms.')
    return out


# ── Constraint parser ───────────────────────────────────────────

def _parse_constraint(line: str, idx: int) -> ConstraintSpec:
    stripped = line.strip()
    if not stripped:
        raise ModelParseError('Blank constraint.')

    lm = _LABEL_RE.match(stripped)
    if lm:
        name, body = lm.group(1).upper(), lm.group(2).strip()
    else:
        name, body = f'ROW{idx}', stripped

    ops = list(_RELATION_RE.finditer(body))
    if len(ops) != 1:
        raise ModelParseError(
            f'Constraint {name} must have exactly one operator (<, >, <=, >=, =).'
        )
    op_m = ops[0]
    left = body[: op_m.start()].strip()
    op = op_m.group(1)
    right = body[op_m.end() :].strip()
    if not left or not right:
        raise ModelParseError(f'Constraint {name} is incomplete.')

    lt, lc = _parse_expr(left)
    rt, rc = _parse_expr(right)

    for v, c in rt.items():
        lt[v] = lt.get(v, 0.0) - c
    lt = {v: c for v, c in lt.items() if abs(c) > EPSILON}
    if not lt:
        raise ModelParseError(f'Constraint {name} has no variable terms.')

    if op == '<':
        op = '<='
    elif op == '>':
        op = '>='

    return ConstraintSpec(name=name, lhs_terms=lt, operator=op, rhs=rc - lc, source=body)


# ── Post-END declarations ──────────────────────────────────────

def _parse_post_end(lines: list[str], spec: ModelSpec) -> None:
    for raw in lines:
        line = _strip_comment(raw).strip()
        if not line:
            continue
        toks = line.split()
        kw = toks[0].upper()

        if kw == 'FREE' and len(toks) >= 2:
            spec.free_vars.add(toks[1].upper())
        elif kw == 'GIN' and len(toks) >= 2:
            spec.integer_vars.add(toks[1].upper())
        elif kw == 'INT' and len(toks) >= 2:
            spec.binary_vars.add(toks[1].upper())
        elif kw == 'SLB' and len(toks) >= 3:
            try:
                spec.lower_bounds[toks[1].upper()] = float(toks[2])
            except ValueError:
                raise ModelParseError(f'Invalid SLB value: {toks[2]}')
        elif kw == 'SUB' and len(toks) >= 3:
            try:
                spec.upper_bounds[toks[1].upper()] = float(toks[2])
            except ValueError:
                raise ModelParseError(f'Invalid SUB value: {toks[2]}')
        elif kw == 'TITLE':
            spec.title = ' '.join(toks[1:]) if len(toks) > 1 else None
        else:
            raise ModelParseError(f'Unrecognised post-END statement: {line}')


# ── Top-level model parser ─────────────────────────────────────

def parse_model_text(model_text: str) -> ModelSpec:
    # Strip comments, drop blank lines
    lines = [_strip_comment(l).strip() for l in model_text.splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        raise ModelParseError('No model was entered.')

    # Objective
    obj = _detect_obj(lines[0])
    if obj is None:
        raise ModelParseError(
            'First line must begin with MAX, MIN, MAXIMIZE, MINIMIZE, MAXIMISE, or MINIMISE.'
        )
    sense, obj_text = obj
    if not obj_text:
        raise ModelParseError(f"{lines[0].split()[0].upper()} requires an expression on the same line.")

    # Locate END
    end_idx = None
    for i, ln in enumerate(lines):
        if _norm_kw(ln) == 'END':
            end_idx = i
            break
    if end_idx is None:
        raise ModelParseError('Model must end with END.')

    body = lines[:end_idx]
    if len(body) < 2:
        raise ModelParseError('Need objective, ST, at least one constraint, and END.')
    if _norm_kw(body[1]) != 'ST':
        raise ModelParseError('Second line must be ST (SUBJECT TO / SUCH THAT / S.T.).')

    obj_terms = _parse_obj_expr(obj_text)
    con_lines = body[2:]
    if not con_lines:
        raise ModelParseError('At least one constraint is required after ST.')

    constraints: list[ConstraintSpec] = []
    seen: set[str] = set()
    all_vars: set[str] = set(obj_terms)

    for i, cl in enumerate(con_lines, 1):
        c = _parse_constraint(cl, i)
        if c.name in seen:
            raise ModelParseError(f"Duplicate constraint name '{c.name}'.")
        seen.add(c.name)
        constraints.append(c)
        all_vars.update(c.lhs_terms)

    spec = ModelSpec(
        sense=sense,
        objective_terms=obj_terms,
        constraints=constraints,
        variables=sorted(all_vars, key=_natural_key),
    )

    # Post-END declarations
    post = lines[end_idx + 1 :]
    if post:
        _parse_post_end(post, spec)
        for s in (spec.free_vars, spec.integer_vars, spec.binary_vars):
            all_vars.update(s)
        all_vars.update(spec.lower_bounds)
        all_vars.update(spec.upper_bounds)
        spec.variables = sorted(all_vars, key=_natural_key)

    return spec


# ── Solver ──────────────────────────────────────────────────────

def _cbc():
    if pulp is None:
        raise RuntimeError('PuLP is not installed. Install with: python3 -m pip install pulp')
    s = pulp.PULP_CBC_CMD(msg=False)
    if not s.available():
        raise RuntimeError('CBC solver is not available in this environment.')
    return s


def solve_model(spec: ModelSpec) -> SolveResult:
    try:
        solver = _cbc()
    except RuntimeError as e:
        return SolveResult('SolverUnavailable', None, {}, [], str(e))

    sense = pulp.LpMaximize if spec.sense == 'MAX' else pulp.LpMinimize
    prob = pulp.LpProblem(spec.title or 'Korado_Model', sense)

    # Build variables with LINDO declarations applied
    pv: dict[str, pulp.LpVariable] = {}
    for name in spec.variables:
        lo: float | None = 0       # default: non-negative
        up: float | None = None
        cat = 'Continuous'

        if name in spec.free_vars:
            lo = None               # FREE removes lower bound
        if name in spec.lower_bounds:
            lo = spec.lower_bounds[name]
        if name in spec.upper_bounds:
            up = spec.upper_bounds[name]
        if name in spec.integer_vars:
            cat = 'Integer'
        elif name in spec.binary_vars:
            cat = 'Binary'

        pv[name] = pulp.LpVariable(name, lowBound=lo, upBound=up, cat=cat)

    # Objective
    prob += pulp.lpSum(c * pv[v] for v, c in spec.objective_terms.items())

    # Constraints
    for con in spec.constraints:
        expr = pulp.lpSum(c * pv[v] for v, c in con.lhs_terms.items())
        if con.operator == "<=":
            prob += expr <= con.rhs, con.name
        elif con.operator == ">=":
            prob += expr >= con.rhs, con.name
        else:
            prob += expr == con.rhs, con.name

    try:
        prob.solve(solver)
    except Exception as e:
        return SolveResult('SolverError', None, {}, [], str(e))

    status = pulp.LpStatus.get(prob.status, 'Undefined')
    if status != 'Optimal':
        return SolveResult(status, None, {}, [], None)

    vals = {n: pv[n].value() for n in spec.variables}
    crs: list[ConstraintResult] = []
    for con in spec.constraints:
        lv = sum((vals[v] or 0.0) * c for v, c in con.lhs_terms.items())
        if con.operator == "<=":
            sl = con.rhs - lv
        elif con.operator == ">=":
            sl = lv - con.rhs
        else:
            sl = abs(lv - con.rhs)
        sl = _near_zero(sl)
        crs.append(ConstraintResult(con.name, sl, abs(sl) <= EPSILON, con.source))

    return SolveResult("Optimal", pulp.value(prob.objective), vals, crs)


# ── Output ──────────────────────────────────────────────────────

def print_solution(result: SolveResult, model_text: str, title: str | None = None) -> None:
    width = shutil.get_terminal_size().columns
    print('—' * width + '\n')
    if title:
        print(f' TITLE: {title}\n')

    print(' FORMULATION\n')
    for line in model_text.splitlines():
        s = line.strip()
        if s:
            print(f'  {s}')
    print()

    if result.status == 'Optimal':
        width = shutil.get_terminal_size().columns
        print('—' * width + '\n')
        print(' LP OPTIMUM FOUND')
        print('\n OBJECTIVE FUNCTION VALUE\n')
        print(f'   ={_fmt(result.objective_value)}')
        print(f"\n {'VARIABLE':<16} {'VALUE':>14}")
        for vn, val in result.variable_values.items():
            print(f' {vn:<16} {_fmt(val)}')
        width = shutil.get_terminal_size().columns
        print('\n' + '—' * width)
        print('\n BINDING CONSTRAINTS')
        bind = [r for r in result.constraints if r.binding]
        if not bind:
            print(' NONE')
        else:
            for r in bind:
                print(f" {r.name:<12}SLACK = {_fmt(r.slack).strip():>10}   {r.source}")
    elif result.status == 'Infeasible':
        print(' NO FEASIBLE SOLUTION FOUND')
    elif result.status == 'Unbounded':
        print(' UNBOUNDED SOLUTION')
    elif result.status == 'SolverUnavailable':
        print(' SOLVER IS NOT AVAILABLE')
        if result.message:
            print(f'\n {result.message}')
    elif result.status == 'SolverError':
        print(' SOLVER ERROR')
        if result.message:
            print(f'\n {result.message}')
    else:
        print(f' NO SOLUTION ({result.status})')
        if result.message:
            print(f'\n {result.message}')


# ── Editor helpers ──────────────────────────────────────────────

def _default_lines() -> list[str]:
    return ['']


def _safe_add(win, y, x, text, w, attr=0):
    if y < 0 or x < 0 or w <= 0:
        return
    try:
        win.addnstr(y, x, text, w, attr)
    except curses.error:
        pass


def _has_end_above(lines: list[str], row: int) -> bool:
    """Return True if any line at or above *row* is END."""
    for r in range(row, -1, -1):
        if _norm_kw(lines[r]) == 'END':
            return True
    return False


# ── Curses editor ───────────────────────────────────────────────

def _render(scr, lines, cr, cc, top, cursor_attr):
    scr.erase()
    h, w = scr.getmaxyx()

    hdr = [
        '88      a8P                                            88              ',
        "88    ,88'                                             88              ",
        '88  ,88"                                               88              ',
        "88,d88'      ,adPPYba,  8b,dPPYba, ,adPPYYba,  ,adPPYb,88  ,adPPYba,  ",
        '8888"88,    a8"     "8a 88P\'   "Y8 ""     `Y8 a8"    `Y88 a8"     "8a ',
        '88P   Y8b   8b       d8 88         ,adPPPPP88 8b       88 8b       d8  ',
        '88     "88, "8a,   ,a8" 88         88,    ,88 "8a,   ,d88 "8a,   ,a8"  ',
        '88       Y8b `"YbbdP"\'  88         `"8bbdP"Y8  `"8bbdP"Y8  `"YbbdP"\'  By: Batu Koray Masak  ',
        '',
        'Koray\'s Operations Research App for Decision Optimization',
        '',
        '↑↓←→ Navigate    Enter: new row    Ctrl+G: Solve    Ctrl+Q: Quit ! comments    FREE/GIN/INT/SLB/SUB after END    Non-negativity default',
        ''
    ]

    for i, t in enumerate(hdr):
        if i >= h:
            break
        _safe_add(scr, i, 0, t, w - 1, curses.A_BOLD if i == 0 else 0)

    bs = len(hdr) + 1
    vis = max(1, h - bs - 1)
    if cr < top:
        top = cr
    if cr >= top + vis:
        top = cr - vis + 1

    gw = max(2, len(str(max(1, len(lines)))))

    for vi in range(vis):
        li = top + vi
        sy = bs + vi
        if li >= len(lines):
            _safe_add(scr, sy, 0, '~', 1, curses.A_DIM)
            continue
        pf = f'{li+1:>{gw}}| '
        aw = max(1, w - len(pf) - 1)
        txt = lines[li]
        lo = max(0, cc - aw + 1) if li == cr else 0
        ct = txt[lo: lo + aw]

        is_cur_line = li == cr

        # Render line number
        _safe_add(scr, sy, 0, pf, len(pf), 0)

        # Render line content
        if not is_cur_line:
            _safe_add(scr, sy, len(pf), ct, aw, 0)
        else:
            _safe_add(scr, sy, len(pf), ct, aw, 0)


    try:
        scr.addnstr(h - 1, 0, ' ' * (w - 1), w - 1, curses.A_REVERSE)
        scr.addnstr(h - 1, 1, f' Ln {cr+1}/{len(lines)}  Col {cc+1} ', w - 2, curses.A_REVERSE)
    except curses.error:
        pass

    pf2 = f'{cr+1:>{gw}}| '
    aw2 = max(1, w - len(pf2) - 1)
    lo2 = max(0, cc - aw2 + 1)
    cy = bs + cr - top
    cx = len(pf2) + min(cc - lo2, aw2 - 1)
    try:
        scr.move(cy, cx)
    except curses.error:
        pass
    scr.refresh()
    return top


def _edit_curses(initial: list[str] | None = None) -> str | None:
    lines = list(initial) if initial else _default_lines()

    def _submit():
        return '\n'.join(l.rstrip() for l in lines if l.strip())

    def session(scr):
        curses.curs_set(1)
        scr.keypad(True)

        # Initialize colors for the custom cursor
        curses.start_color()
        curses.use_default_colors()
        # Pair 1: Black text on a white background (for the cursor)
        curses.init_pair(1, 135, -1)
        cursor_attr = curses.color_pair(1)

        row, col, top = 0, len(lines[0]) if lines else 0, 0

        while True:
            if not lines:
                lines.append('')
                row = col = 0
            col = max(0, min(col, len(lines[row])))
            top = _render(scr, lines, row, col, top, cursor_attr)
            key = scr.getch()

            # ── Quit / Submit ───────────────────────────────
            if key == 17:               # Ctrl+Q
                return None
            if key == 3:                # Ctrl+C
                raise KeyboardInterrupt
            if key == 7:                # Ctrl+G  → solve ('Go')
                return _submit()

            # ── Navigation ──────────────────────────────────
            if key == curses.KEY_UP:
                row = max(0, row - 1)
                col = min(col, len(lines[row]))
                continue
            if key == curses.KEY_DOWN:
                row = min(len(lines) - 1, row + 1)
                col = min(col, len(lines[row]))
                continue
            if key == curses.KEY_LEFT:
                if col > 0:
                    col -= 1
                elif row > 0:
                    row -= 1; col = len(lines[row])
                continue
            if key == curses.KEY_RIGHT:
                if col < len(lines[row]):
                    col += 1
                elif row < len(lines) - 1:
                    row += 1; col = 0
                continue
            if key in (curses.KEY_HOME, 1):
                col = 0; continue
            if key in (curses.KEY_END, 5):
                col = len(lines[row]); continue

            # ── Editing ─────────────────────────────────────
            if key in (curses.KEY_BACKSPACE, 127, 8):
                if col > 0:
                    lines[row] = lines[row][:col-1] + lines[row][col:]
                    col -= 1
                elif row > 0:
                    col = len(lines[row-1])
                    lines[row-1] += lines[row]
                    del lines[row]; row -= 1
                continue
            if key == curses.KEY_DC:
                if col < len(lines[row]):
                    lines[row] = lines[row][:col] + lines[row][col+1:]
                elif row < len(lines) - 1:
                    lines[row] += lines[row+1]; del lines[row+1]
                continue

            # ── Enter ───────────────────────────────────────
            if key in (10, 13, curses.KEY_ENTER):
                # On END line → submit immediately
                if _norm_kw(lines[row]) == 'END':
                    return _submit()
                # Blank line after END → submit
                if not lines[row].strip() and _has_end_above(lines, row):
                    return _submit()
                # Split line at cursor: left stays, right moves to new line below
                left = lines[row][:col]
                right = lines[row][col:]
                lines[row] = left
                lines.insert(row + 1, right)
                row += 1
                col = 0
                continue

            if key == 9:  # Tab
                lines[row] = lines[row][:col] + '    ' + lines[row][col:]
                col += 4; continue
            if 32 <= key <= 126:
                lines[row] = lines[row][:col] + chr(key) + lines[row][col:]
                col += 1

    return curses.wrapper(session)


# ── Plain (non-TTY) editor ──────────────────────────────────────

def _edit_plain(initial: list[str] | None = None) -> str | None:
    print('\nEnter a LINDO-style model. Finish with END on its own line.')
    print('Type Q on an empty prompt to quit.\n')
    if initial:
        for l in initial:
            if l.strip():
                print(f'[template] {l}')
        print()
    buf: list[str] = []
    while True:
        try:
            line = input('> ')
        except EOFError:
            return None
        if not line.strip() and not buf:
            continue
        if not buf and line.strip().upper() in {'Q', 'QUIT', 'EXIT'}:
            return None
        buf.append(line)
        if _norm_kw(line) == 'END':
            # Allow optional post-END input
            print('  (Enter post-END declarations, or press Enter on blank line to solve)')
            while True:
                try:
                    pline = input('> ')
                except EOFError:
                    break
                if not pline.strip():
                    break
                buf.append(pline)
            return '\n'.join(buf)


def _edit(initial: list[str] | None = None) -> str | None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _edit_curses(initial)
        except curses.error:
            pass
    return _edit_plain(initial)


# ── Acquire + solve loop helpers ────────────────────────────────

def _acquire(initial: list[str] | None = None) -> tuple[ModelSpec, str] | None:
    cur = list(initial) if initial else _default_lines()
    while True:
        text = _edit(cur)
        if text is None:
            return None
        try:
            return parse_model_text(text), text
        except ModelParseError as e:
            print(f'\nINPUT ERROR: {e}')
            try:
                input('Press Enter to continue editing...')
            except EOFError:
                return None
            cur = text.splitlines() if text.strip() else _default_lines()


def _prompt() -> str:
    while True:
        try:
            a = input('\n[Enter] New model  |  [E] Edit this model  |  [Q] Quit: ').strip().upper()
        except EOFError:
            return 'quit'
        if not a:
            return 'new'
        if a in {'E', 'EDIT'}:
            return 'edit'
        if a in {'Q', 'QUIT', 'EXIT'}:
            return 'quit'


def _run_example() -> tuple[ModelSpec, str]:
    text = '\n'.join(EXAMPLE_MODEL_LINES)
    spec = parse_model_text(text)
    print_solution(solve_model(spec), text, spec.title)
    return spec, text


# ── Main ────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Korado — Koray's Operations Research App for Decision Optimization.")
    ap.add_argument('--example', action='store_true',
                     help='Solve the built-in example model immediately.')
    args = ap.parse_args(argv)

    last: list[str] | None = None

    if args.example:
        _spec, mt = _run_example()
        last = mt.splitlines()
        act = _prompt()
        if act == 'quit':
            return 0
        if act != 'edit':
            last = None

    while True:
        got = _acquire(initial=last)
        if got is None:
            return 0
        spec, mt = got
        last = mt.splitlines()
        print_solution(solve_model(spec), mt, spec.title)
        act = _prompt()
        if act == 'quit':
            return 0
        if act == 'new':
            last = None


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(0)
