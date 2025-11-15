import unittest
import logging
from toponetx.classes.cell import Cell
import numpy as np
from topology.cell.featured_cell_complex import FeaturedCellComplex
from topology.cell.featured_cell import FeaturedCell
from topology.complex_laplacian import ComplexLaplacian, CellType
from topology import LaplacianType


class FeaturedCellComplexTest(unittest.TestCase):
    @unittest.skip('Ignored')
    def test_init_1(self):
        featured_cells = [FeaturedCell(Cell(elements=[1, 2], rank=1)),
                          FeaturedCell(Cell(elements=[1, 3], rank=1)),
                          FeaturedCell(Cell(elements=[3, 2], rank=1))]
        featured_cell_complex = FeaturedCellComplex(featured_cells)
        logging.info(featured_cell_complex)
        self.assertTrue(True)

    @unittest.skip('Ignored')
    def test_init_2(self):
        try:
            featured_cells = [FeaturedCell.build(indices=[1, 2], rank=1, features=np.array([2.0, 1.4])),
                              FeaturedCell.build(indices=[1, 3], rank=1, features=np.array([1.0, 0.4])),
                              FeaturedCell.build(indices=[3, 2], rank=1, features=np.array([0.8, 1.1]))]
            featured_cell_complex = FeaturedCellComplex(featured_cells)
            logging.info(featured_cell_complex)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_adjacency_matrix(self):
        featured_cell_complex = FeaturedCellComplexTest.generate_featured_cell_complex()
        A = featured_cell_complex.adjacency_matrix()
        logging.info(f'\nAdjacency Matrix:\n{A}')
        self.assertTrue(True)

    def test_co_adjacency_matrix(self):
        featured_cell_complex = FeaturedCellComplexTest.generate_featured_cell_complex()
        A = featured_cell_complex.co_adjacency_matrix()
        logging.info(f'\nCo Adjacency Matrix:\n{A}')

    def test_incidence_matrix(self):
        try:
            featured_cell_complex = FeaturedCellComplexTest.generate_featured_cell_complex()
            logging.info(featured_cell_complex)
            B = featured_cell_complex.incidence_matrix()
            logging.info(f'\nIncidence Matrix\n{B}')
        except (ValueError, TypeError) as e:
            logging.error(e)
            self.assertFalse(False)

    # @unittest.skip('Ignored')
    def test_up_laplacian(self):
        try:
            featured_cell_complex = FeaturedCellComplexTest.generate_featured_cell_complex()
            logging.info(featured_cell_complex)
            simplicial_laplacian_0 = ComplexLaplacian[CellType](laplacian_type=LaplacianType.UpLaplacian,
                                                                rank=0,
                                                                signed=True)
            up_laplacian_rk0 = featured_cell_complex.laplacian(simplicial_laplacian_0)
            logging.info(f'\nUP Laplacian Rank 0:\n{up_laplacian_rk0}')
        except (TypeError, ValueError) as e:
            logging.error(e)
            self.assertFalse(False)

    def test_exercise_hodge_laplacian(self):
        try:
            edges = [[1, 2], [1, 4], [1, 5], [2, 3], [3, 6], [4, 6], [4, 5]]
            cells_2 = [[1, 2, 3, 6, 4], [1, 4, 5]]

            featured_edges = [FeaturedCell.build(indices=edge, rank=1) for edge in edges]
            features_cells_2 = [FeaturedCell.build(indices=face, rank=2) for face in cells_2]
            featured_cells = featured_edges + features_cells_2
            featured_cell_complex = FeaturedCellComplex(featured_cells)
            complex_laplacian = ComplexLaplacian[CellType](laplacian_type=LaplacianType.HodgeLaplacian,
                                                           rank=1,
                                                           signed=True)
            hodge_laplacian_rk1 = featured_cell_complex.laplacian(complex_laplacian)
            logging.info(f'\nHodge Laplacian Rank 1:\n{hodge_laplacian_rk1}')
        except (TypeError, ValueError) as e:
            logging.error(e)
            self.assertFalse(False)


    @staticmethod
    def generate_featured_cell_complex() -> FeaturedCellComplex:
        edges = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        cells_2 = [[2, 3, 4], [1, 2, 3], [1, 3, 4, 5]]

        featured_edges = [FeaturedCell.build(indices=edge, rank=1) for edge in edges]
        features_cells_2 = [FeaturedCell.build(indices=face, rank=2) for face in cells_2]
        featured_cells = featured_edges + features_cells_2
        return FeaturedCellComplex(featured_cells)




