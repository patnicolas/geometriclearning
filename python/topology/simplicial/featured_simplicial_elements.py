__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Python standard library imports
from typing import AnyStr, List, Self
from dataclasses import dataclass
# Library imports
from topology.simplicial.featured_simplex import FeaturedSimplex
__all__ = ['FeaturedSimplicialElements']


@dataclass
class FeaturedSimplicialElements:
    """
    Wrapper for the complex elements with simplex_2 = {faces, cell}
        Simplicial=> nodes, edges, faces
        Cell => nodes, edges, cells

    @param featured_nodes List of complex elements associated with graph node
    @param featured_edges List of complex elements associated with graph edges
    @param featured_faces List of complex elements associated with face or cells
    """
    featured_nodes: List[FeaturedSimplex]     # 0-Simplex
    featured_edges: List[FeaturedSimplex]     # 1-Simplex
    featured_faces: List[FeaturedSimplex]     # 2-Simplex

    @classmethod
    def build(cls, featured_simplex: List[List[FeaturedSimplex]]) -> Self:
        """
        Alternative constructor for building a set of complex elements for a given graph

        @param featured_simplex: List of list of complex elements
        @type featured_simplex:  List[List[FeaturedSimplex]]
        @return: Instance of GraphComplexElements
        @rtype: FeaturedSimplicialElements
        """
        assert len(featured_simplex) == 3, f'The number of sets of complex elements {len(featured_simplex)} should be 3'
        return cls(featured_simplex[0], featured_simplex[1], featured_simplex[2])

    def add_nodes(self, featured_nodes: List[FeaturedSimplex]) -> None:
        self.featured_nodes = featured_nodes

    def add_edges(self, featured_edges: List[FeaturedSimplex]) -> None:
        self.featured_edges = featured_edges

    def add_simplex_2(self, featured_faces: List[FeaturedSimplex]) -> None:
        self.featured_faces = featured_faces

    def dump(self, count: int) -> AnyStr:
        """
        Generate a dump of the set of complex elements
        @param count: Number of complex elements associated with nodes, edges and simplex_2 to be dump
        @type count: int
        @return: Dump of complex elements
        @rtype: str
        """
        assert count > 0, f'Count {count} to display complex elements associated with a graph should be >0'

        nodes_elements_str = '\n'.join([str(s) for s in self.featured_nodes[:count]])
        edges_elements_str = '\n'.join([str(s) for s in self.featured_edges[:count]])
        faces_elements_str = '\n'.join([str(s) for s in self.featured_faces[:count]])
        return (f'\n{len(self.featured_nodes)} Complex nodes\n{nodes_elements_str}\n...'
                f'\n{len(self.featured_edges)} Complex edges\n{edges_elements_str}\n...'
                f'\n{len(self.featured_faces)} Complex faces\n{faces_elements_str}\n...')
