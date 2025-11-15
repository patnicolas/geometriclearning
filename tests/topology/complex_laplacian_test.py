import unittest
from topology.complex_laplacian import ComplexLaplacian, SimplexType, CellType
from topology.cell.featured_cell import FeaturedCell
from topology import LaplacianType
import logging
import python


class ComplexLaplacianTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_simplicial_up_laplacian(self):
        try:
            simplicial_laplacian = ComplexLaplacian[SimplexType](laplacian_type=LaplacianType.UpLaplacian,
                                                                 rank=1,
                                                                 signed=True)
            logging.info(simplicial_laplacian)
            edges = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            faces = [[1, 2, 3], [2, 3, 4]]
            up_laplacian = simplicial_laplacian(edges + faces)
            logging.info(f'Simplicial Up Laplacian:\n{up_laplacian}')
            self.assertTrue(True)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignored')
    def test_simplicial_down_laplacian(self):
        try:
            simplicial_laplacian = ComplexLaplacian[SimplexType](laplacian_type=LaplacianType.DownLaplacian,
                                                                 rank=0,
                                                                 signed=True)
            edges = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            faces = [[1, 2, 3], [2, 3, 4]]
            down_laplacian = simplicial_laplacian(edges + faces)
            logging.info(f'{down_laplacian}')
            self.assertTrue(False)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skip('Ignored')
    def test_simplicial_down_laplacian_2(self):
        try:
            simplicial_laplacian = ComplexLaplacian[SimplexType](laplacian_type=LaplacianType.DownLaplacian,
                                                                 rank=2,
                                                                 signed=True)
            edges = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            faces = [[1, 2, 3], [2, 3, 4]]
            down_laplacian = simplicial_laplacian(edges + faces)
            logging.info(f'Simplicial Down Laplacian:\n{down_laplacian}')
            self.assertTrue(True)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignored')
    def test_cell_down_laplacian_2(self):
        try:
            cell_laplacian = ComplexLaplacian[CellType](laplacian_type=LaplacianType.DownLaplacian,
                                                        rank=2,
                                                        signed=True)
            edges_indices = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
            cell2_elements = [[1, 2, 3], [2, 3, 4, 5]]
            featured_edges = [FeaturedCell.build(indices=edge, rank=1).cell for edge in edges_indices]
            featured_cell2s = [FeaturedCell.build(indices=face, rank=2).cell for face in cell2_elements]
            down_laplacian = cell_laplacian(featured_edges + featured_cell2s)
            logging.info(f'Cell Down Laplacian:\n{down_laplacian}')
            self.assertTrue(True)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignored')
    def test_simplicial_hodge_laplacian(self):
        edges = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5]]
        faces = [[1, 2, 3], [2, 3, 4]]

        try:
            for n in range(1, 3):
                simplicial_laplacian = ComplexLaplacian[SimplexType](laplacian_type=LaplacianType.HodgeLaplacian,
                                                                     rank=1,
                                                                     signed=True)
                hodge_laplacian = simplicial_laplacian(edges + faces)
                logging.info(f'Simplicial Hodge Laplacian:\n{hodge_laplacian}')
            self.assertTrue(True)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(False)

    #@unittest.skip('Ignored')
    def test_cell_hodge_laplacian(self):
        edges = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
        cell2s = [[2, 3, 4], [1, 2, 3], [1, 3, 4, 5]]

        try:
            featured_edges = [FeaturedCell.build(indices=edge, rank=1).cell for edge in edges]
            featured_cell2s = [FeaturedCell.build(indices=face, rank=2).cell for face in cell2s]
            featured_cells = featured_edges + featured_cell2s
            cell_laplacian = ComplexLaplacian[CellType](laplacian_type=LaplacianType.UpLaplacian,
                                                        rank=1,
                                                        signed=True)
            up_laplacian = cell_laplacian(featured_cells)
            logging.info(f'Cell Up Laplacian rank 1:\n{up_laplacian}')
            cell_laplacian = ComplexLaplacian[CellType](laplacian_type=LaplacianType.DownLaplacian,
                                                        rank=2,
                                                        signed=True)
            down_laplacian = cell_laplacian(featured_cells)
            logging.info(f'Cell Down Laplacian rank 2:\n{down_laplacian}')
            for n in range(0, 3):
                cell_laplacian = ComplexLaplacian[CellType](laplacian_type=LaplacianType.HodgeLaplacian,
                                                            rank=n,
                                                            signed=True)
                hodge_laplacian = cell_laplacian(featured_cells)
                logging.info(f'Cell Hodge Laplacian rank {n}:\n{hodge_laplacian}')
            self.assertTrue(True)
        except (ValueError, KeyError, TypeError) as e:
            logging.error(e)
            self.assertTrue(False)
