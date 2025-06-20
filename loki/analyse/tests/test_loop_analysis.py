# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine
from loki.frontend import available_frontends
from loki.analyse import get_loop_tree
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_tree(frontend):
    fcode = """
subroutine driver(a, b, c, m, n, p)
  real, intent(inout) :: a(:,:,:), b(:,:,:), c(:,:,:)
  integer, intent(in) :: m, n, p
  integer             :: i,j,k

  do i=1,m

    do j=1,n
      do k=1,p
        b(k,j,i) = k
      end do
    end do

    do j=2,n
      a(:,j,i) = j
    end do
  end do

  do i=2,m
    c(:,:,i) = i
  end do

  do i=3,m
    do j=3,m
      do k=2,p
        c(k,j,i) = i+j+k
      end do
    end do
  end do
end subroutine driver
"""
    driver = Subroutine.from_source(fcode, frontend=frontend)

    loops = FindNodes(ir.Loop).visit(driver.ir)
    assert len(loops) == 8

    loop_tree = get_loop_tree(driver.ir)
    assert len(loop_tree.roots) == 3

    # verify that pre-order dfs walk order is corect
    for tree_node, loop in zip(loop_tree.walk_depth_first(), loops):
        assert tree_node.loop == loop, "pre order dfs walk should be the same sequence as FindNodes"

    # verify that loop map works correctly
    for tree_node, loop in zip(loop_tree.walk_depth_first(), loops):
        assert tree_node == loop_tree.get_tree_node(loop), "wrong loop_map entry for curent loop"

    # verify that depths of loop tree nodes are correct
    depths = [0, 1, 2, 1, 0, 0, 1, 2]
    for tree_node, depth in zip(loop_tree.walk_depth_first(), depths):
        assert tree_node.depth == depth

    # verify that post-order dfs walk order is correct
    post_order_indices = [2, 1, 3, 0, 4, 7, 6, 5]
    post_order_loops = [loops[i] for i in post_order_indices]
    for tree_node, loop in zip(loop_tree.walk_depth_first(pre_order=False), post_order_loops):
        assert tree_node.loop == loop

    # verify that bfs walk order is correct
    bfs_indices = [0, 4, 5, 1, 3, 6, 2, 7]
    bfs_loops = [loops[i] for i in bfs_indices]
    for tree_node, loop in zip(loop_tree.walk_breadth_first(), bfs_loops):
        assert tree_node.loop == loop

    # verify that depths are correct with dfs walk
    bfs_depths = [0, 0, 0, 1, 1, 1, 2, 2]
    for tree_node, depth in zip(loop_tree.walk_breadth_first(), bfs_depths):
        assert tree_node.depth == depth
