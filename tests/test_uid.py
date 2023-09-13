# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from collections import defaultdict
import pytest
from loki import Sourcefile
from loki.visitors.visitor import Visitor
from loki.tools import is_iterable, as_tuple


@pytest.fixture(scope="module", name="here")
def fixture_here():
    return Path(__file__).parent


class Checker(Visitor):
    def __init__(self):
        super().__init__()
        self.uid_collection = set()

    def visit_all(self, item, *args, **kwargs):
        if is_iterable(item) and not args:
            return as_tuple(self.visit(i, **kwargs) for i in item if i is not None)
        return as_tuple(self.visit(i, **kwargs) for i in [item, *args] if i is not None)

    def visit_Module(self, o, **kwargs):
        self.visit(o.spec, **kwargs)
        self.visit_all(o.contains, **kwargs)

    def visit_Subroutine(self, o, **kwargs):
        self.visit(o.docstring, **kwargs)
        self.visit(o.spec, **kwargs)
        self.visit(o.body, **kwargs)
        self.visit_all(o.contains, **kwargs)

    def visit_Node(self, o, **kwargs):
        assert hasattr(o, "_uid")

        uid = o._uid
        assert uid is not None
        if uid in self.uid_collection:
            raise AssertionError("uid is not unique")
        self.uid_collection.add(uid)
        self.visit_all(o.children, **kwargs)

    def visit_tuple(self, o, **kwargs):
        return self.visit_all(o, **kwargs)

    visit_list = visit_tuple


fortran_files = list(Path(".").rglob("*.[fF]90"))


@pytest.mark.parametrize("file", fortran_files)
def test_has_every_node_a_uid(here: Path, file: Path):
    print(here, file)
    try:
        source = Sourcefile.from_file(file)
    except Exception:
        try:
            source = Sourcefile.from_file(file, preprocess=True)
        except Exception:
            return  # ignore if a file cannot be parsed

    Checker().visit(source.ir)


loki_module = __import__("loki")
ir_module = loki_module.ir
node_types = [getattr(ir_module, node_type) for node_type in ir_module.__all__]

constructor_args = {node_type: [] for node_type in node_types}

dummy_scalar = loki_module.Scalar("dummy")
dummy_variable = loki_module.Variable(name="dummy")
dummy_comment = loki_module.Comment("dummy")
dummy_int = loki_module.IntLiteral(42)

constructor_args = defaultdict(
    list,
    {
        ir_module.Associate: [
            (
                (
                    loki_module.Array("dummy", None, None, (dummy_scalar,)),
                    dummy_scalar,
                ),
            )
        ],
        ir_module.Loop: [
            dummy_scalar,
            loki_module.Range(("dummy", 2, 4)),
        ],
        ir_module.WhileLoop: [None],
        ir_module.Conditional: [loki_module.Comparison(dummy_scalar, "<=", dummy_int)],
        ir_module.Assignment: [
            dummy_scalar,
            dummy_int,
        ],
        ir_module.CallStatement: [
            dummy_variable,
            (dummy_int,),
        ],
        ir_module.Allocation: [(dummy_variable,)],
        ir_module.Deallocation: [(dummy_variable,)],
        ir_module.Nullify: [(dummy_variable,)],
        ir_module.Comment: ["dummy"],
        ir_module.CommentBlock: [(dummy_comment,)],
        ir_module.Pragma: ["dummy"],
        ir_module.Import: [None],
        ir_module.VariableDeclaration: [(dummy_variable,)],
        ir_module.ProcedureDeclaration: [(dummy_variable,)],
        ir_module.DataDeclaration: [
            (dummy_variable,),
            (dummy_int,),
        ],
        ir_module.StatementFunction: [
            dummy_variable,
            (dummy_int,),
            dummy_variable,
            loki_module.SymbolAttributes("int"),
        ],
        ir_module.MultiConditional: [
            dummy_scalar,
            ((dummy_int,),),
            ((loki_module.Intrinsic("dummy"),),),
            (loki_module.Intrinsic("dummy"),),
        ],
        ir_module.MaskedStatement: [
            (
                loki_module.Comparison(dummy_scalar, "<", dummy_int),
                loki_module.Comparison(dummy_scalar, ">", dummy_int),
            ),
            (
                (ir_module.Assignment(dummy_scalar, dummy_int),),
                (ir_module.Assignment(dummy_scalar, dummy_int),),
            ),
            (ir_module.Assignment(dummy_scalar, dummy_int),),
        ],
        ir_module.Intrinsic: ["dummy"],
        ir_module.Enumeration: [(dummy_variable,)],
        ir_module.RawSource: ["dummy"],
    },
)

@pytest.mark.parametrize("node_type", node_types)
def test_update_retains_uid(node_type):
    args = constructor_args[node_type]
    node = node_type(*args)
    old_uid = node._uid
    node._update(*args)
    assert old_uid == node._uid

@pytest.mark.parametrize("node_type", node_types)
def test_clone_changes_uid(node_type):
    args = constructor_args[node_type]
    node = node_type(*args)
    new_node = node.clone()
    assert node._uid is not new_node._uid
