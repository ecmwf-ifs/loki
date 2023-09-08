# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from os import remove
from pathlib import Path
import pytest
from conftest import graphviz_present
from loki import Sourcefile
from loki.visitors.pretty_visualize import pretty_visualize, Visualizer


@pytest.fixture(scope="module", name="here")
def fixture_here():
    return Path(__file__).parent


test_files = [
    "trivial_fortran_files/case_statement_subroutine.f90",
    "trivial_fortran_files/if_else_statement_subroutine.f90",
    "trivial_fortran_files/module_with_subroutines.f90",
    "trivial_fortran_files/nested_if_else_statements_subroutine.f90",
]

solutions = {
    "trivial_fortran_files/case_statement_subroutine.f90": {
        "node_count": 12,
        "edge_count": 11,
        "node_labels": {
            "0": "<Section::>",
            "1": "<Subroutine:: check_grade>",
            "2": "<Section::>",
            "3": "<VariableDeclaration:: score>",
            "4": "<Section::>",
            "5": "<MultiConditional:: score>",
            "6": "<Intrinsic:: PRINT *, A>",
            "7": "<Intrinsic:: PRINT *, B>",
            "8": "<Intrinsic:: PRINT *, C>",
            "9": "<Intrinsic:: PRINT *, D>",
            "10": "<Intrinsic:: PRINT *, F>",
            "11": "<Intrinsic:: PRINT *, Inv...>",
        },
        "connectivity_list": {
            "0": ["1"],
            "1": ["2", "4"],
            "2": ["3"],
            "4": ["5"],
            "5": ["6", "7", "8", "9", "10", "11"],
        },
    },
    "trivial_fortran_files/if_else_statement_subroutine.f90": {
        "node_count": 8,
        "edge_count": 7,
        "node_labels": {
            "0": "<Section::>",
            "1": "<Subroutine:: check_number>",
            "2": "<Section::>",
            "3": "<VariableDeclaration:: x>",
            "4": "<Section::>",
            "5": "x > 0.0",
            "6": "<Intrinsic:: PRINT *, The...>",
            "7": "<Intrinsic:: PRINT *, The...>",
        },
        "connectivity_list": {
            "0": ["1"],
            "1": ["2", "4"],
            "2": ["3"],
            "4": ["5"],
            "5": ["6", "7"],
        },
    },
    "trivial_fortran_files/module_with_subroutines.f90": {
        "node_count": 22,
        "edge_count": 21,
        "node_labels": {
            "0": "<Section::>",
            "1": "<Module:: math_operations>",
            "2": "<Section::>",
            "3": "<Intrinsic:: IMPLICIT NONE>",
            "4": "<Subroutine:: add>",
            "5": "<Section::>",
            "6": "<VariableDeclaration:: x, y>",
            "7": "<VariableDeclaration:: result>",
            "8": "<Section::>",
            "9": "<Assignment:: result = x + y>",
            "10": "<Subroutine:: subtract>",
            "11": "<Section::>",
            "12": "<VariableDeclaration:: x, y>",
            "13": "<VariableDeclaration:: result>",
            "14": "<Section::>",
            "15": "<Assignment:: result = x - y>",
            "16": "<Subroutine:: multiply>",
            "17": "<Section::>",
            "18": "<VariableDeclaration:: x, y>",
            "19": "<VariableDeclaration:: result>",
            "20": "<Section::>",
            "21": "<Assignment:: result = x*y>",
        },
        "connectivity_list": {
            "0": ["1"],
            "1": ["2", "4", "10", "16"],
            "10": ["11", "14"],
            "11": ["12", "13"],
            "14": ["15"],
            "16": ["17", "20"],
            "17": ["18", "19"],
            "2": ["3"],
            "20": ["21"],
            "4": ["5", "8"],
            "5": ["6", "7"],
            "8": ["9"],
        },
    },
    "trivial_fortran_files/nested_if_else_statements_subroutine.f90": {
        "node_count": 12,
        "edge_count": 11,
        "node_labels": {
            "0": "<Section::>",
            "1": "<Subroutine:: nested_if_example>",
            "2": "<Section::>",
            "3": "<VariableDeclaration:: x, y>",
            "4": "<Section::>",
            "5": "x > 0",
            "6": "y > 0",
            "7": "<Intrinsic:: PRINT *, Bot...>",
            "8": "<Intrinsic:: PRINT *, x i...>",
            "9": "y > 0",
            "10": "<Intrinsic:: PRINT *, x i...>",
            "11": "<Intrinsic:: PRINT *, Bot...>",
        },
        "connectivity_list": {
            "0": ["1"],
            "1": ["2", "4"],
            "2": ["3"],
            "4": ["5"],
            "5": ["6", "9"],
            "6": ["7", "8"],
            "9": ["10", "11"],
        },
    },
}


def get_property(node_edge_info, name):
    for node_info, edge_info in node_edge_info:
        if name in node_info and name in edge_info:
            yield (node_info[name], edge_info[name])
            continue

        if name in node_info:
            yield (node_info[name], None)
            continue

        if name in edge_info:
            yield (None, edge_info[name])
            continue

        if node_info and edge_info:
            raise KeyError(f"Keyword {name} not found!")


@pytest.mark.skipif(not graphviz_present(), reason="Graphviz is not installed")
@pytest.mark.parametrize("test_file", test_files)
def test_Visualizer(here, test_file):
    solution = solutions[test_file]
    source = Sourcefile.from_file(here / test_file)

    visualizer = Visualizer()
    node_edge_info = [item for item in visualizer.visit(source.ir) if item is not None]

    node_names = [name for (name, _) in get_property(node_edge_info, "name")]
    node_labels = [label for (label, _) in get_property(node_edge_info, "label")]

    assert len(node_names) == len(node_labels) == solution["node_count"]

    for name, label in zip(node_names, node_labels):
        assert solution["node_labels"][name] == label

    edge_heads = [head for (_, head) in get_property(node_edge_info, "head_name")]
    edge_tails = [tail for (_, tail) in get_property(node_edge_info, "tail_name")]

    assert len(edge_heads) == len(edge_tails) == solution["edge_count"]

    for head, tail in zip(edge_heads, edge_tails):
        assert head in solution["connectivity_list"][tail]


@pytest.mark.skipif(not graphviz_present(), reason="Graphviz is not installed")
@pytest.mark.parametrize("test_file", test_files)
def test_pretty_visualize_can_write_graphs(here, test_file):
    source = Sourcefile.from_file(here / test_file)

    name = "test_pretty_visualize_write_graphs"
    source.ir.view(visualization=True, filename=here / name)

    path = Path(here / name)
    assert path.is_file()
    remove(path)

    path = Path(here / (name + ".pdf"))
    assert path.is_file()
    remove(path)


def find_edges(input_text):
    pattern = re.compile(r"(\d+)\s*->\s*(\d+)", re.IGNORECASE)
    return re.findall(pattern, input_text)


def find_nodes(input_text):
    pattern = re.compile(r'\d+ *\[[^\[\]]*(?:"[^"]*"[^\[\]]*)*\]', re.IGNORECASE)
    return re.findall(pattern, input_text)


def find_node_id_inside_nodes(input_text):
    pattern = re.compile(r"(\d+)\s+\[", re.IGNORECASE)
    return re.findall(pattern, input_text)


def find_label_content_inside_nodes(input_text):
    pattern = re.compile(r'label="([^"]*"|\'[^\']*\'|[^\'"]*)"', re.IGNORECASE)
    return re.findall(pattern, input_text)


@pytest.mark.skipif(not graphviz_present(), reason="Graphviz is not installed")
@pytest.mark.parametrize("test_file", test_files)
def test_pretty_visualize_writes_correct_graphs(here, test_file):
    solution = solutions[test_file]
    source = Sourcefile.from_file(here / test_file)

    graph = pretty_visualize(source.ir)

    edges = find_edges(str(graph))

    for start, stop in edges:
        assert stop in solution["connectivity_list"][start]

    nodes = find_nodes(str(graph))

    assert len(edges) == solution["edge_count"]
    assert len(nodes) == solution["node_count"]

    node_ids = [find_node_id_inside_nodes(node) for node in nodes]
    for found_node_id in node_ids:
        assert len(found_node_id) == 1

    found_labels = [find_label_content_inside_nodes(node) for node in nodes]
    for found_label in found_labels:
        assert len(found_label) == 1

    assert len(found_labels) == len(node_ids)

    for node, label in zip(node_ids, found_labels):
        assert solution["node_labels"][node[0]] == label[0]
