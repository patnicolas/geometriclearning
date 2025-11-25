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

# Standard Library imports
from typing import List, Tuple, Self, TypeVar, AnyStr, Dict
# 3rd Party imports
import numpy as np
from toponetx import ColoredHyperGraph
# Library imports
from toponetx.classes.hyperedge import HyperEdge
from topology.hypergraph.featured_hyperedge import FeaturedHyperEdge
from topology.featured_complex import FeaturedComplex
__all__ = ['FeaturedHyperGraph']
T = TypeVar('T')


class FeaturedHyperGraph(FeaturedComplex[T]):
    """
    Implementation of featured hypergraph:
    This implementation leverages the class toponetx.ColoredHypergraph. A hypergraph is a generalization of a graph
    where a hyperedge can connect any number of nodes or vertices. Similarly to Cell and Simplicial Complexes, a node
    and a hyperedge are said to be incident if the vertex is a member of the hyperedge.
    The parameter type 'T' is the type of the first element of the representation of the components of this topological
    domain - Simplicial -> List[int],  Cell -> Cell, Hypergraph -> List[Tuple[int]]

    The Up, Down and Hodge Laplacian matrices are computed by converting the Hypergraph into a Simplicial Complex then
    computing the appropriate Laplacian matrices. Therefore, the simplicial complex dictionary {rank : List of Tuple
    of node indices} is defined as empty and initialized once any of the Laplacian matrices has to be computed.

    Note: Hyperedges can be labeled with any identifier. I use indices (int) start with 1 to stay consistent with
    my implementation of simplicial complex and cell complexes in the GitHub repository.
    """
    def __init__(self, featured_hyperedges: frozenset[FeaturedHyperEdge]) -> None:
        """
        Default constructor for the Featured Hypergraph. The features Hypergraph consists of an immutable list of
        Featured hyperedge defined as toponetx hyperedge + feature vector.

        @param featured_hyperedges: Immutable list of featured hyperedges
        @type featured_hyperedges: frozenset[FeaturedHyperEdge]
        """
        if len(featured_hyperedges) == 0:
            raise ValueError('Cannot instantiate a featured hypergraph without hyperedges')

        super(FeaturedHyperGraph, self).__init__()

        self.featured_hyperedges = featured_hyperedges
        self.simplicial_complex_dict: Dict[int, List[Tuple[int, ...]]] = {}

    def __str__(self) -> AnyStr:
        featured_hyperedges = '\n'.join([str(featured_hyperedge) for featured_hyperedge in self.featured_hyperedges])
        simplicial_elements = ' \n'.join([str(simplex) for simplex in self.simplicial_complex_dict])
        return f'\nHyperedges:\n{featured_hyperedges}\nSimplicial Elements: {simplicial_elements}'

    @classmethod
    def build(cls,
              hyperedge_indices_list: List[Tuple[int, ...]],
              ranks: List[int],
              features_list: List[np.array] = None) -> Self:
        """
        Alternative constructor for the featured hypergraph using a higher granularity descriptor for the hyperedges
        The list of hyperedge node indices, ranks and feature vectors should be identical and a ValueError is raised
        if this condition is not met.

        @param hyperedge_indices_list: List of Tuple of indices
        @type hyperedge_indices_list: List[Tuple[int, ...]]
        @param ranks: List of rank for each hyperedge
        @type ranks: List[int]
        @param features_list: List of features
        @type features_list: List of Numpy array representing
        @return: Instance of FeaturedHyperGraph
        @rtype: FeaturedHyperGraph
        """
        if len(hyperedge_indices_list) != len(features_list):
            raise ValueError(
                f'Num of hyperedges {len(hyperedge_indices_list)} should == num of features vector {len(features_list)}'
            )
        if len(hyperedge_indices_list) != len(ranks):
            raise ValueError(
                f'Num of hyperedges {len(hyperedge_indices_list)} should == num of ranks {len(ranks)}'
            )
        featured_hyperedges = frozenset([FeaturedHyperEdge(hyperedge=HyperEdge(elements=featured_indices, rank=rank),
                                                           features=features)
                                        for featured_indices, rank, features
                                         in zip(hyperedge_indices_list, ranks, features_list)])
        return cls(featured_hyperedges)

    def set_simplicial_complex(self) -> None:
        """
        Instantiate the simplicial complex associated with this Featured Hypergraph. The conversion into a simplicial
        complex is required to compute the various Laplacian matrices.
        """
        from itertools import combinations

        # Generate edges then faces
        for dim in range(1, 3):
            simplices = set[Tuple[int, ...]]()
            for featured_hyperedge in self.featured_hyperedges:
                hyperedge = sorted(featured_hyperedge.hyperedge)
                # If the number of indices exceeds 3 (faces)
                # break the simplex into triangles
                if len(hyperedge) >= dim + 1:
                    for simplex in combinations(hyperedge, dim + 1):
                        simplices.add(simplex)

            self.simplicial_complex_dict[dim] = sorted(simplices)

    def laplacian(self, complex_laplacian: T) -> np.array:
        """
        Compute the Up, Down and Hodge Laplacian matrices for this featured hypergraph. The computation Laplacian
        can only be performed on the associated Simplicial Complex, therefore, the attribute simplicial_complex_dict
        must be initialized prior to any computation.
        The parameter type 'T' is the type of the first element of the representation of this topological domain (List,
        Cell,).

        @param complex_laplacian: Parameterized complex Laplacian configuration which type is related to the specific
                topological domain such as Simplicial Complex or Hypergraph.
        @type complex_laplacian: Parameterized complex
        @return: Laplacian matrix
        @rtype: Numpy array
        """
        from itertools import chain

        # Step 1: Create the equivalent simplicial complex
        if len(self.simplicial_complex_dict) == 0:
            self.set_simplicial_complex()

        # Step 2: Collect the simplices node indices
        collected_simplices_list = list(self.simplicial_complex_dict.values())

        # Step 3: Flatten the list of list of simplices
        simplices = list(chain.from_iterable(collected_simplices_list))
        return complex_laplacian(simplices)

    def adjacency_matrix(self, ranks: Tuple[int, int] | int, signed: bool = False) -> np.array:
        if ranks[0] >= ranks[1]:
            raise ValueError(f'Ranks for adjacency matrix {ranks} should be rank1 < rank2')
        colored_hyper_graph = self.__get_colored_hypergraph()
        return colored_hyper_graph.adjacency_matrix(rank=ranks[0], via_rank=ranks[1]).todense()

    def incidence_matrix(self, ranks: Tuple[int, int] = (0, 1), directed_graph: bool = True) -> np.array:
        if ranks[0] >= ranks[1]:
            raise ValueError(f'Ranks for incidence matrix {ranks} should be rank1 < rank2')

        colored_hyper_graph = self.__get_colored_hypergraph()
        return colored_hyper_graph.incidence_matrix(rank=ranks[0], to_rank=ranks[1]).todense()

    """ -------------------------   Private helper method ---------------------------------- """

    def __get_colored_hypergraph(self) -> ColoredHyperGraph:
        return ColoredHyperGraph([featured_hyperedge.hyperedge
                                  for featured_hyperedge in self.featured_hyperedges])






