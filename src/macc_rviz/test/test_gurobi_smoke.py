"""Gurobi smoke test.

Verifies: gurobipy imports, the *academic* license (not the free restricted
one) activates, a trivial LP solves to the expected objective, and a
>2000-variable model builds without tripping the free-license cap.

Run manually:
  ~/macc-venv/bin/python3 -m pytest test/test_gurobi_smoke.py -v -s
"""

import os
import sys

import pytest  # noqa: I100, I101

gurobipy = pytest.importorskip('gurobipy')


# Gurobi's free restricted license caps models at 2000 vars and 2000
# linear constraints. We build well past that to confirm we're not on it.
_FREE_LIMIT = 2000


def _find_license_file():
    """Return path to the gurobi.lic the runtime would use, or None."""
    env = os.environ.get('GRB_LICENSE_FILE')
    if env and os.path.exists(env):
        return env
    for p in (
        os.path.expanduser('~/gurobi.lic'),
        '/opt/gurobi/gurobi.lic',
        '/Library/gurobi/gurobi.lic',
    ):
        if os.path.exists(p):
            return p
    return None


def _parse_license_fields(path):
    """Return a dict of KEY=VALUE fields from the license file."""
    fields = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                fields[k.strip()] = v.strip()
    return fields


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gurobipy_import_and_version():
    v = gurobipy.gurobi.version()
    print(f'\n[gurobi] gurobipy version: {v}')
    assert isinstance(v, tuple) and len(v) == 3
    # Not asserting exact version — just that it's ≥13 so the MILP APIs work.
    assert v[0] >= 13


def test_license_file_is_academic():
    path = _find_license_file()
    assert path is not None, 'no gurobi.lic found (checked env + ~/gurobi.lic)'
    fields = _parse_license_fields(path)
    print(f'\n[gurobi] license file: {path}')
    print(f'[gurobi] TYPE={fields.get("TYPE")} '
          f'VERSION={fields.get("VERSION")} '
          f'EXPIRATION={fields.get("EXPIRATION")} '
          f'LICENSEID={fields.get("LICENSEID")}')
    assert fields.get('TYPE') == 'ACADEMIC', (
        f'expected ACADEMIC license, got TYPE={fields.get("TYPE")!r}. '
        f'If this is the free restricted license, model size will be '
        f'capped and the MILP work will fail silently at scale.'
    )


def test_env_starts_without_restricted_warning(capsys):
    """Creating an Env must succeed and announce an ACADEMIC license.

    The free restricted license prints 'Restricted license - for '
    'non-production use only' on Env startup; we fail if we see that.
    """
    with gurobipy.Env() as env:
        # Touch a parameter so the env is fully initialised.
        env.setParam('OutputFlag', 0)
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    print(f'\n[gurobi] env startup output:\n{combined.strip() or "(silent)"}')
    assert 'Restricted license' not in combined, (
        'Gurobi reports a restricted license — academic license not loaded.'
    )


def test_trivial_lp_solves_to_expected_objective():
    """min x+y  s.t. x+y>=2, 0<=x,y<=10  →  optimal objective = 2.0"""
    with gurobipy.Env() as env:
        env.setParam('OutputFlag', 0)
        with gurobipy.Model('trivial', env=env) as m:
            x = m.addVar(lb=0, ub=10, name='x')
            y = m.addVar(lb=0, ub=10, name='y')
            m.addConstr(x + y >= 2)
            m.setObjective(x + y, gurobipy.GRB.MINIMIZE)
            m.optimize()
            assert m.Status == gurobipy.GRB.OPTIMAL, f'status={m.Status}'
            assert abs(m.ObjVal - 2.0) < 1e-6, f'ObjVal={m.ObjVal}'
            print(f'\n[gurobi] trivial LP ObjVal={m.ObjVal:.6f}')


def test_model_exceeds_free_restricted_limits():
    """Build an LP with > 2000 vars and > 2000 constraints and solve it.

    On the free restricted license this would fail with a Model size limit
    error. On an academic license it solves cleanly.
    """
    n = _FREE_LIMIT + 500   # comfortably above the cap
    with gurobipy.Env() as env:
        env.setParam('OutputFlag', 0)
        with gurobipy.Model('big', env=env) as m:
            xs = m.addVars(n, lb=0, ub=1, name='x')
            for i in range(n):
                m.addConstr(xs[i] >= 0.5)
            m.setObjective(gurobipy.quicksum(xs[i] for i in range(n)),
                           gurobipy.GRB.MINIMIZE)
            m.optimize()
            assert m.Status == gurobipy.GRB.OPTIMAL, f'status={m.Status}'
            print(f'\n[gurobi] big LP n={n} '
                  f'NumVars={m.NumVars} NumConstrs={m.NumConstrs} '
                  f'ObjVal={m.ObjVal:.3f}')
            assert m.NumVars > _FREE_LIMIT
            assert m.NumConstrs > _FREE_LIMIT


def test_banner_summary():
    """Prints a one-shot banner for the log record."""
    v = '.'.join(str(x) for x in gurobipy.gurobi.version())
    path = _find_license_file()
    fields = _parse_license_fields(path) if path else {}
    print(
        f'\n[gurobi] smoke OK — version={v} '
        f'TYPE={fields.get("TYPE")} '
        f'EXPIRATION={fields.get("EXPIRATION")} '
        f'python={sys.version.split()[0]}'
    )
