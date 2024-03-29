# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from pathlib import Path
import pytest
from conftest import graphviz_present
from loki import Sourcefile
from loki.analyse import dataflow_analysis_attached
from loki.ir import Node, FindNodes, ir_graph, GraphCollector


@pytest.fixture(scope="module", name="here")
def fixture_here():
    return Path(__file__).parent


test_files = [
    "sources/trivial_fortran_files/case_statement_subroutine.f90",
    "sources/trivial_fortran_files/if_else_statement_subroutine.f90",
    "sources/trivial_fortran_files/module_with_subroutines.f90",
    "sources/trivial_fortran_files/nested_if_else_statements_subroutine.f90",
]

solutions_default_parameters = {
    "sources/trivial_fortran_files/case_statement_subroutine.f90": {
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
    "sources/trivial_fortran_files/if_else_statement_subroutine.f90": {
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
    "sources/trivial_fortran_files/module_with_subroutines.f90": {
        "node_count": 24,
        "edge_count": 23,
        "node_labels": {
            "0": "<Section::>",
            "1": "<Module:: math_operations>",
            "2": "<Section::>",
            "3": "<Intrinsic:: IMPLICIT NONE>",
            "4": "<Section::>",
            "5": "<Intrinsic:: CONTAINS>",
            "6": "<Subroutine:: add>",
            "7": "<Section::>",
            "8": "<VariableDeclaration:: x, y>",
            "9": "<VariableDeclaration:: result>",
            "10": "<Section::>",
            "11": "<Assignment:: result = x + y>",
            "12": "<Subroutine:: subtract>",
            "13": "<Section::>",
            "14": "<VariableDeclaration:: x, y>",
            "15": "<VariableDeclaration:: result>",
            "16": "<Section::>",
            "17": "<Assignment:: result = x - y>",
            "18": "<Subroutine:: multiply>",
            "19": "<Section::>",
            "20": "<VariableDeclaration:: x, y>",
            "21": "<VariableDeclaration:: result>",
            "22": "<Section::>",
            "23": "<Assignment:: result = x*y>",
        },
        "connectivity_list": {
            "0": ["1"],
            "1": ["2", "4"],
            "10": ["11"],
            "12": ["13", "16"],
            "13": ["14", "15"],
            "16": ["17"],
            "18": ["19", "22"],
            "19": ["20", "21"],
            "2": ["3"],
            "22": ["23"],
            "4": ["5", "6", "12", "18"],
            "6": ["7", "10"],
            "7": ["8", "9"],
        },
    },
    "sources/trivial_fortran_files/nested_if_else_statements_subroutine.f90": {
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

solutions_node_edge_counts = {
    "sources/trivial_fortran_files/case_statement_subroutine.f90": {
        "node_count": [[12, 19], [14, 21]],
        "edge_count": [[11, 18], [13, 20]],
    },
    "sources/trivial_fortran_files/if_else_statement_subroutine.f90": {
        "node_count": [[8, 9], [10, 11]],
        "edge_count": [[7, 8], [9, 10]],
    },
    "sources/trivial_fortran_files/module_with_subroutines.f90": {
        "node_count": [[24, 39], [32, 47]],
        "edge_count": [[23, 38], [31, 46]],
    },
    "sources/trivial_fortran_files/nested_if_else_statements_subroutine.f90": {
        "node_count": [[12, 14], [14, 16]],
        "edge_count": [[11, 13], [13, 15]],
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
@pytest.mark.parametrize("show_comments", [True, False])
@pytest.mark.parametrize("show_expressions", [True, False])
def test_graph_collector_node_edge_count_only(
    here, test_file, show_comments, show_expressions
):
    solution = solutions_node_edge_counts[test_file]
    source = Sourcefile.from_file(here / test_file)

    graph_collector = GraphCollector(
        show_comments=show_comments, show_expressions=show_expressions
    )
    node_edge_info = [
        item for item in graph_collector.visit(source.ir) if item is not None
    ]

    node_names = [name for (name, _) in get_property(node_edge_info, "name")]
    node_labels = [label for (label, _) in get_property(node_edge_info, "label")]

    assert (
        len(node_names)
        == len(node_labels)
        == solution["node_count"][show_comments][show_expressions]
    )
    edge_heads = [head for (_, head) in get_property(node_edge_info, "head_name")]
    edge_tails = [tail for (_, tail) in get_property(node_edge_info, "tail_name")]

    assert (
        len(edge_heads)
        == len(edge_tails)
        == solution["edge_count"][show_comments][show_expressions]
    )


@pytest.mark.skipif(not graphviz_present(), reason="Graphviz is not installed")
@pytest.mark.parametrize("test_file", test_files)
def test_graph_collector_detail(here, test_file):
    solution = solutions_default_parameters[test_file]
    source = Sourcefile.from_file(here / test_file)

    graph_collector = GraphCollector()
    node_edge_info = [
        item for item in graph_collector.visit(source.ir) if item is not None
    ]

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
@pytest.mark.parametrize("linewidth", [40, 60, 80])
def test_graph_collector_maximum_label_length(here, test_file, linewidth):
    source = Sourcefile.from_file(here / test_file)

    graph_collector = GraphCollector(
        show_comments=True, show_expressions=True, linewidth=linewidth
    )
    node_edge_info = [
        item for item in graph_collector.visit(source.ir) if item is not None
    ]
    node_labels = [label for (label, _) in get_property(node_edge_info, "label")]

    for label in node_labels:
        assert len(label) <= linewidth


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
def test_ir_graph_writes_correct_graphs(here, test_file):
    solution = solutions_default_parameters[test_file]
    source = Sourcefile.from_file(here / test_file)

    graph = ir_graph(source.ir)

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


@pytest.mark.parametrize("test_file", test_files)
def test_ir_graph_dataflow_analysis_attached(here, test_file):
    source = Sourcefile.from_file(here / test_file)

    def find_lives_defines_uses(text):
        # Regular expression pattern to match content within square brackets after 'live:', 'defines:', and 'uses:'
        pattern = r"live:\s*\[([^\]]*?)\],\s*defines:\s*\[([^\]]*?)\],\s*uses:\s*\[([^\]]*?)\]"
        matches = re.search(pattern, text)
        assert matches

        def remove_spaces_and_newlines(text):
            return text.replace(" ", "").replace("\n", "")

        def disregard_empty_strings(elements):
            return set(element for element in elements if element != "")

        def apply_conversion(text):
            return disregard_empty_strings(remove_spaces_and_newlines(text).split(","))

        return (
            apply_conversion(matches.group(1)),
            apply_conversion(matches.group(2)),
            apply_conversion(matches.group(3)),
        )

    for routine in source.all_subroutines:
        with dataflow_analysis_attached(routine):
            for node in FindNodes(Node).visit(routine.body):
                node_info, _ = GraphCollector(show_comments=True).visit(node)[0]
                lives, defines, uses = find_lives_defines_uses(node_info["label"])
                assert node.live_symbols == set(lives)
                assert node.uses_symbols == set(uses)
                assert node.defines_symbols == set(defines)
