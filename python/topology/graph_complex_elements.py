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
from topology.simplicial.abstract_simplicial_complex import ComplexElement


@dataclass
class GraphComplexElements:
    """
    Wrapper for the complex elements with simplex_2 = {faces, cell}
        Simplicial=> nodes, edges, faces
        Cell => nodes, edges, cells

    @param complex_nodes List of complex elements associated with graph node
    @param complex_edges List of complex elements associated with graph edges
    @param complex_simplex_2 List of complex elements associated with face or cells
    """
    complex_nodes: List[ComplexElement]
    complex_edges: List[ComplexElement]
    complex_simplex_2: List[ComplexElement]

    @classmethod
    def build(cls, complex_elements: List[List[ComplexElement]]) -> Self:
        """
        Alternative constructor for building a set of complex elements for a given graph

        @param complex_elements: List of list of complex elements
        @type complex_elements:  List[List[ComplexElement]]
        @return: Instance of GraphComplexElements
        @rtype: GraphComplexElements
        """
        assert len(complex_elements) == 3, f'The number of sets of complex elements {len(complex_elements)} should be 3'
        return cls(complex_elements[0], complex_elements[1], complex_elements[2])

    def dump(self, count: int) -> AnyStr:
        """
        Generate a dump of the set of complex elements
        @param count: Number of complex elements associated with nodes, edges and simplex_2 to be dump
        @type count: int
        @return: Dump of complex elements
        @rtype: str
        """
        assert count > 0, f'Count {count} to display complex elements associated with a graph should be >0'

        nodes_elements_str = '\n'.join([str(s) for s in self.complex_nodes[:count]])
        edges_elements_str = '\n'.join([str(s) for s in self.complex_edges[:count]])
        faces_elements_str = '\n'.join([str(s) for s in self.complex_simplex_2[:count]])
        return (f'\n{len(self.complex_nodes)} Complex nodes\n{nodes_elements_str}\n...'
                f'\n{len(self.complex_edges)} Complex edges\n{edges_elements_str}\n...'
                f'\n{len(self.complex_simplex_2)} Complex faces\n{faces_elements_str}\n...')
