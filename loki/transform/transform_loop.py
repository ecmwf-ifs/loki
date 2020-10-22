"""
Collection of utility routines that provide loop transformations.

"""
from loki.ir import Loop
from loki.tools import is_loki_pragma, flatten
from loki.visitors import FindNodes, Transformer

__all__ = ['fuse_loops']


def fuse_loops(routine):
    """
    Searches for loops annotated with the `loki fuse` pragma and attempts
    to fuse them into a single loop.
    """
    loop_list = [loop for loop in FindNodes(Loop).visit(routine.body)
                 if is_loki_pragma(loop.pragma, starts_with='fuse')]
    if not loop_list:
        return

    variable = loop_list[0].variable
    bounds = loop_list[0].bounds
    body = flatten(loop.body for loop in loop_list)
    loop_map = {loop_list[0]: Loop(variable=variable, body=body, bounds=bounds)}
    loop_map.update({loop: None for loop in loop_list[1:]})
    routine.body = Transformer(loop_map).visit(routine.body)
