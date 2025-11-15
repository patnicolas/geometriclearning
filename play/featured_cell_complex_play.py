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
import logging
import python
# 3rd party library imports
from toponetx.classes.cell import Cell
import toponetx as tnx
# Library imports
from play import Play
from topology.cell.featured_cell_complex import FeaturedCellComplex
from topology.cell.featured_cell import FeaturedCell
from topology.complex_laplacian import ComplexLaplacian, CellType
from topology import LaplacianType


class FeaturedCellComplexPlay(Play):
    """
      Wrapper to implement the evaluation of cell complexes as defined in Substack article:
      "Graphs Reimagined: The Power of Cell Complexes"

      References:
      - Article: https://patricknicolas.substack.com/p/exploring-simplicial-complexes-for
      - FeaturedCellComplex:
          https://github.com/patnicolas/geometriclearning/blob/main/python/topology/simplicial/abstract_simplicial_complex.py


      The features are implemented by the class FeaturedCellComplex in the source file
                    python/topology/cell/featured_cell_complex.py
      The class FeaturedCellComplexPlay is a wrapper of the class FeaturedCellComplex
      The execution of the tests follows the same order as in the Substack article
      """
    def __init__(self) -> None:
        super(FeaturedCellComplexPlay, self).__init__()
        self.featured_cell_complex = FeaturedCellComplexPlay.generate_featured_cell_complex()
        logging.info(self.featured_cell_complex)

    def play(self) -> None:
        FeaturedCellComplexPlay.play_attributes()
        self.play_laplacians()
        self.play_adjacency()
        self.play_incidence()

    def play_adjacency(self) -> None:
        """
        Implementation of the evaluation code for the Substack article "Graphs Reimagined: The Power of Cell Complexes"
        - Code snippets 4 & 5
        """
        A = self.featured_cell_complex.adjacency_matrix()
        logging.info(f'\nAdjacency Matrix:\n{A}')
        A = self.featured_cell_complex.co_adjacency_matrix()
        logging.info(f'\nCo Adjacency Matrix:\n{A}')

    def play_incidence(self) -> None:
        """
        Implementation of the evaluation code for the Substack article "Graphs Reimagined: The Power of Cell Complexes"
        Code snippets 7
        """
        B = self.featured_cell_complex.incidence_matrix()
        logging.info(f'\nIncidence Matrix\n{B}')

    def play_laplacians(self) -> None:
        """
        Implementation of the evaluation code for the Substack article "Graphs Reimagined: The Power of Cell Complexes"
        Code snippets 9
        """
        logging.info(self.featured_cell_complex)
        cell_laplacian = ComplexLaplacian[CellType](laplacian_type=LaplacianType.UpLaplacian,
                                                    rank=1,
                                                    signed=True)
        up_laplacian_rk1 = self.featured_cell_complex.laplacian(cell_laplacian)
        logging.info(f'\nUP Laplacian Rank 0:\n{up_laplacian_rk1}')
        cell_laplacian = ComplexLaplacian[CellType](laplacian_type=LaplacianType.DownLaplacian,
                                                    rank=2,
                                                    signed=True)
        down_laplacian_rk2 = self.featured_cell_complex.laplacian(cell_laplacian)
        logging.info(f'Cell Down Laplacian rank 2:\n{down_laplacian_rk2}')

        for rank in range(0, 3):
            cell_laplacian = ComplexLaplacian[CellType](laplacian_type=LaplacianType.HodgeLaplacian,
                                                        rank=rank,
                                                        signed=True)
            hodge_laplacian = self.featured_cell_complex.laplacian(cell_laplacian)
            logging.info(f'Cell Hodge Laplacian rank {rank}:\n{hodge_laplacian}')

    @staticmethod
    def play_attributes() -> None:
        """
        Implementation of the evaluation code for the Substack article "Graphs Reimagined: The Power of Cell Complexes"
        Appendix
        """
        # Generate cells
        edge_indices = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        cell_2_indices = [[2, 3, 4], [1, 2, 3], [1, 3, 4, 5]]
        edges = [Cell(elements=edge_index, rank=1) for edge_index in edge_indices]
        cell_2_s = [Cell(elements=cell_2_index, rank=2) for cell_2_index in cell_2_indices]
        # Instantiate the cell complex
        cx = tnx.CellComplex(edges + cell_2_s)

        # Add features to edges
        cx.cells[edge_indices[0]]['color'] = 'blue'
        cx.cells[edge_indices[0]]['name'] = 'Pontiac'
        cx.cells[edge_indices[0]]['year'] = 2013
        cx.cells[edge_indices[1]]['color'] = 'red'
        cx.cells[edge_indices[1]]['name'] = 'Testarossa'
        cx.cells[edge_indices[1]]['year'] = 1988

        # Add features to Cell rank 2
        cx.cells[cell_2_indices[0]]['brand'] = 'General Motor'
        cx.cells[cell_2_indices[0]]['country'] = 'USA'
        cx.cells[cell_2_indices[1]]['brand'] = 'Ferrari'
        cx.cells[cell_2_indices[1]]['country'] = 'Italy'
        logging.info({k: v for k, v in cx.cells[edge_indices[0]].items()})
        logging.info({k: v for k, v in cx.cells[edge_indices[1]].items()})
        logging.info({k: v for k, v in cx.cells[cell_2_indices[0]].items()})
        logging.info({k: v for k, v in cx.cells[cell_2_indices[1]].items()})

    @staticmethod
    def generate_featured_cell_complex() -> FeaturedCellComplex:
        edges_indices = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        cells_2_indices = [[2, 3, 4], [1, 2, 3], [1, 3, 4, 5]]
        featured_edges = [FeaturedCell.build(indices=edge_index, rank=1) for edge_index in edges_indices]
        features_cells_2 = [FeaturedCell.build(indices=cell_2, rank=2) for cell_2 in cells_2_indices]
        return FeaturedCellComplex(featured_edges + features_cells_2)


if __name__ == '__main__':
    featured_cell_complex_play = FeaturedCellComplexPlay()
    featured_cell_complex_play.play_attributes()
