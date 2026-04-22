"""Regression test: launch file must coerce non-string params via ParameterValue.

Bug history: ``LaunchConfiguration("x")`` is a Substitution that resolves
to a string at runtime. When passed bare into a Node action's
``parameters=[{...}]`` list for a node parameter declared as bool / int /
float, ROS2's parameter system silently rejects the type-mismatched
override and the node falls back to its declared default. This made all
non-string CLI overrides via ``ros2 launch`` silently inert (e.g.
``use_example_structure:=true`` → still random structure;
``seed:=42`` → still fresh random seed).

Fix: wrap each non-string LaunchConfiguration in
``ParameterValue(LaunchConfiguration(name), value_type=...)`` so the
substitution is coerced to the correct ROS parameter type before
dispatch to the node.

This test introspects the LaunchDescription returned by
``generate_launch_description()`` and asserts that every parameter dict
in the Node's ``parameters`` list maps to either a literal Python value
(e.g. ``step_duration_sec=0.4``), a bare LaunchConfiguration (only for
string-typed node params), or a ParameterValue with an explicit
``value_type`` matching the node-side declaration.
"""

import importlib.util
import os

from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


LAUNCH_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'launch', 'macc.launch.py')
)


# Map: param name -> expected python type the Node side expects.
# Mirrors macc_rviz_sim.declare_parameter() declarations.
EXPECTED_TYPES = {
    'num_robots': int,
    'use_example_structure': bool,
    'seed': int,
    'grid_x': int,
    'grid_y': int,
    'grid_z': int,
    'density': float,
    'cbs_max_t': int,
    'cbs_branch_limit': int,
    'milp_per_t_time_limit': float,
    'milp_total_time_limit': float,
    'milp_mip_gap': float,
    'milp_T_max': int,
}

# Params allowed to be passed as bare LaunchConfiguration (string-typed
# on the node side, so substitution-as-string round-trips correctly).
STRING_PARAMS = {'planner'}


def _load_launch_module():
    spec = importlib.util.spec_from_file_location('macc_launch', LAUNCH_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_node_action(launch_description):
    for entity in launch_description.entities:
        if isinstance(entity, Node):
            return entity
    raise AssertionError('No Node action found in LaunchDescription')


def _key_to_str(key):
    """Resolve a parameter dict key to its string name.

    launch_ros normalises ``parameters=[{"name": value}]`` by wrapping
    each key in a tuple of ``TextSubstitution`` objects. Unwrap them.
    """
    from launch.substitutions import TextSubstitution
    if isinstance(key, str):
        return key
    if isinstance(key, tuple) and len(key) == 1:
        inner = key[0]
        if isinstance(inner, TextSubstitution):
            return inner.text
        if isinstance(inner, str):
            return inner
    if isinstance(key, TextSubstitution):
        return key.text
    raise AssertionError(f'Unexpected param key shape: {key!r}')


def _flatten_param_dicts(node_action):
    """Return one merged {name: value} dict from the Node's parameters list."""
    # Node stores parameters under a name-mangled attribute. Try both the
    # mangled name and the public-ish name to be robust to launch_ros API
    # tweaks across distros.
    params_list = getattr(
        node_action,
        '_Node__parameters',
        getattr(node_action, 'parameters', None),
    )
    assert params_list is not None, (
        'Could not access Node parameters list (ros2 launch_ros internal '
        'API may have changed).'
    )
    merged = {}
    for entry in params_list:
        if isinstance(entry, dict):
            for k, v in entry.items():
                merged[_key_to_str(k)] = v
    return merged


def _value_type_of(parameter_value):
    """Return the declared value_type of a ParameterValue, across API revs."""
    return getattr(
        parameter_value,
        'value_type',
        getattr(parameter_value, '_ParameterValue__value_type', None),
    )


def test_launch_param_types_coerced():
    """Every non-string launch param must be wrapped in ParameterValue.

    Pre-fix this fails: 9 of 10 launch params are passed as bare
    LaunchConfiguration substitutions, which ROS2 silently rejects on
    type mismatch when handed to bool/int/float node parameters.
    """
    mod = _load_launch_module()
    ld = mod.generate_launch_description()
    node = _get_node_action(ld)
    params = _flatten_param_dicts(node)

    def _unwrap(value):
        # launch_ros normalises some single-Substitution values into a
        # 1-tuple of Substitutions; unwrap to compare the actual object.
        if isinstance(value, tuple) and len(value) == 1:
            return value[0]
        return value

    failures = []
    for name, expected_type in EXPECTED_TYPES.items():
        assert name in params, f'launch file is missing param {name!r}'
        value = _unwrap(params[name])
        if not isinstance(value, ParameterValue):
            failures.append(
                f'{name}: expected ParameterValue(..., '
                f'value_type={expected_type.__name__}), got bare '
                f'{type(value).__name__} — ROS2 will silently drop this'
            )
            continue
        actual_type = _value_type_of(value)
        if actual_type is not expected_type:
            failures.append(
                f'{name}: ParameterValue value_type={actual_type}, '
                f'expected {expected_type.__name__}'
            )

    # String params should be passed as bare LaunchConfiguration (no
    # coercion needed; this also exercises the path that we explicitly
    # do NOT want to over-wrap).
    for name in STRING_PARAMS:
        assert name in params, f'launch file is missing param {name!r}'
        value = _unwrap(params[name])
        assert isinstance(value, LaunchConfiguration), (
            f'{name} (string) should be a bare LaunchConfiguration; '
            f'got {type(value).__name__}'
        )

    assert not failures, (
        'Launch file dropped non-string params '
        '(silent ROS2 type-mismatch):\n  ' + '\n  '.join(failures)
    )


def test_launch_param_values_propagate():
    """End-to-end: non-default grid/density overrides evaluate to typed values.

    The static test above only checks that each param is wrapped in
    ParameterValue(value_type=T). This one simulates a CLI override
    (``grid_x:=7 grid_y:=7 grid_z:=4 density:=0.42``) by seeding the
    LaunchContext's launch configurations, evaluating the resolved node
    parameters, and asserting the final Python types and values match.
    """
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    mod = _load_launch_module()
    ld = mod.generate_launch_description()

    ctx = LaunchContext()
    overrides = {
        'grid_x': '7',
        'grid_y': '7',
        'grid_z': '4',
        'density': '0.42',
    }
    # Apply declared-arg defaults first, then layer the CLI-style overrides
    # on top (mirrors how ros2 launch resolves argument precedence).
    for entity in ld.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.visit(ctx)
    for k, v in overrides.items():
        ctx.launch_configurations[k] = v

    node = _get_node_action(ld)
    params = _flatten_param_dicts(node)

    def _evaluate(pv):
        # ParameterValue exposes an evaluate(context) method that runs
        # the internal substitutions and applies the declared value_type.
        return pv.evaluate(ctx)

    expectations = {
        'grid_x': (int, 7),
        'grid_y': (int, 7),
        'grid_z': (int, 4),
        'density': (float, 0.42),
    }
    for name, (expected_type, expected_value) in expectations.items():
        pv = params[name]
        if isinstance(pv, tuple) and len(pv) == 1:
            pv = pv[0]
        evaluated = _evaluate(pv)
        assert type(evaluated) is expected_type, (
            f'{name}: evaluated type {type(evaluated).__name__}, '
            f'expected {expected_type.__name__}'
        )
        assert evaluated == expected_value, (
            f'{name}: evaluated value {evaluated!r}, '
            f'expected {expected_value!r}'
        )
