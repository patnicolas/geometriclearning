import unittest
import logging
from toponetx.classes.cell import Cell
import numpy as np
from topology.cell.featured_cell_complex import FeaturedCellComplex
from topology.cell.featured_cell import FeaturedCell
from topology.complex_laplacian import ComplexLaplacian
from topology import LaplacianType


class FeaturedCellComplexTest(unittest.TestCase):
    def test_init_1(self):
        cells = [FeaturedCell(Cell(elements=[1, 2], rank=1)),
                 FeaturedCell(Cell(elements=[1, 3], rank=1)),
                 FeaturedCell(Cell(elements=[3, 2], rank=1))]
        featured_cell_complex = FeaturedCellComplex(cells)
        logging.info(featured_cell_complex)
        self.assertTrue(True)

    def test_init_2(self):
        try:
            cells = [FeaturedCell.build(indices=[1, 2], rank=1, features=np.array([2.0, 1.4])),
                     FeaturedCell.build(indices=[1, 3], rank=1, features=np.array([1.0, 0.4])),
                     FeaturedCell.build(indices=[3, 2], rank=1, features=np.array([0.8, 1.1]))]
            featured_cell_complex = FeaturedCellComplex(cells)
            logging.info(featured_cell_complex)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_adjacency_matrix(self):
        f_cells_2 = [FeaturedCell(Cell(elements=[1, 2], rank=1)),
                     FeaturedCell(Cell(elements=[1, 3], rank=1)),
                     FeaturedCell(Cell(elements=[3, 2], rank=1))]

        featured_cell_complex = FeaturedCellComplex(f_cells_2)
        A = featured_cell_complex.adjacency_matrix()
        logging.info(f'\nAdjacency Matrix:\n{A}')
        self.assertTrue(True)

    def test_incidence_matrix(self):
        try:
            cells = [FeaturedCell.build(indices=[1, 2], rank=1, features=np.array([2.0, 1.4])),
                     FeaturedCell.build(indices=[1, 3], rank=1, features=np.array([1.0, 0.4])),
                     FeaturedCell.build(indices=[3, 2], rank=1, features=np.array([0.8, 1.1])),
                     FeaturedCell.build(indices=[2, 4], rank=1, features=np.array([2.8, 0.1])),
                     FeaturedCell.build(indices=[1, 2, 4], rank=2)
                     ]

            featured_cell_complex = FeaturedCellComplex(cells)
            logging.info(featured_cell_complex)
            B = featured_cell_complex.incidence_matrix()
            logging.info(f'\nIncidence Matrix\n{B}')
        except ValueError as e:
            logging.error(e)
            self.assertFalse(False)

    def test_up_laplacian(self):
        edges = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        faces = [[2, 3, 4], [1, 2, 3], [1, 3, 4, 5]]
        edge_cells = [FeaturedCell.build(indices=edge, rank=1) for edge in edges]
        face_cells = [FeaturedCell.build(indices=face, rank=2) for face in faces]
        cells = edge_cells + face_cells
        featured_cell_complex = FeaturedCellComplex(cells)
        logging.info(featured_cell_complex)

        simplicial_laplacian_0 = ComplexLaplacian(laplacian_type=LaplacianType.UpLaplacian,
                                                  rank=0,
                                                  signed=True)
        up_laplacian_rk0 = featured_cell_complex.laplacian(simplicial_laplacian_0)
        logging.info(f'\nUP Laplacian Rank 0:\n{up_laplacian_rk0}')


