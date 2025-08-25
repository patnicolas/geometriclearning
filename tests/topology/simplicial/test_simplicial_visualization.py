import unittest
import logging
from topology.simplicial.abstract_simplicial_complex import AbstractSimplicialComplex
from topology.simplicial.simplicial_visualization import SimplicialVisualization


class SimplicialVisualizationTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_show_1(self):
        edge_set = [
            [1, 2], [1, 3], [1, 6], [2, 3], [5, 3], [3, 6], [3, 4], [1, 7], [6, 7], [7, 8], [8, 9], [7, 9], [5, 6],
            [4, 5], [3, 6]
        ]
        face_set = [
            [1, 2, 3], [5, 3, 6], [3, 5, 4], [1, 6, 7], [7, 8, 9],  # Triangles
            [5, 3, 6, 4]
        ]
        try:
            logging.info(f'Number of edges: {len(edge_set)}')
            logging.info(f'Number of faces: {len(face_set)}')
            font_attributes = {'title_font_size': 18, 'face_font_size': 15, 'feature_font_size': 10 }
            simplicial_visualization = SimplicialVisualization(
                AbstractSimplicialComplex.random(4, edge_set, face_set),
                font_attributes
            )
            simplicial_visualization.show()
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_show_4(self):
        edge_set = [
            [1, 2], [1, 3], [1, 6], [2, 3], [5, 3], [3, 6], [3, 4], [1, 7], [6, 7], [7, 8], [8, 9], [7, 9], [5, 6],
            [4, 5], [3, 6], [9, 10], [9, 7], [10, 11], [7, 11], [7, 10], [11, 12], [7, 12], [7, 13], [13, 14], [14, 15],
            [15, 16], [16, 17], [17, 18], [7, 14], [7, 16], [18, 19], [19, 20], [20, 21], [1, 16], [1, 17], [17, 21],
            [19, 21], [21, 22], [22, 23], [23, 1], [1, 21], [1, 22]
        ]
        face_set = [
            [1, 2, 3], [5, 3, 6], [3, 5, 4], [1, 6, 7], [7, 8, 9], [7, 9, 10], [7, 10, 11], [7, 13, 12], [7, 14, 16],
            [7, 13, 14], [1, 16, 17], [17, 21, 19], [17, 21, 1], [21, 22, 1],  # Triangles
            [5, 3, 6, 4], [7, 9, 10, 11], [7, 13, 14, 16], [1, 17, 19, 21]  # Tetrahedrons
        ]
        try:
            logging.info(f'Number of edges: {len(edge_set)}')
            logging.info(f'Number of faces: {len(face_set)}')
            font_attributes = {'title_font_size': 18, 'face_font_size': 15, 'feature_font_size': 10 }
            simplicial_visualization = SimplicialVisualization(
                AbstractSimplicialComplex.random(3, edge_set, face_set),
                font_attributes
            )
            simplicial_visualization.show()
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)


@unittest.skip('Ignore')
def test_show_2(self):
    edge_set = [
        [1, 2], [2, 3], [5, 6], [3, 4], [1, 6], [4, 5], [3, 6], [4, 6], [6, 7], [7, 1], [2, 6]
    ]
    face_set = [
        [4, 3, 6], [6, 2, 3], [5, 4, 6], [6, 7, 1],  # Triangle
        [6, 4, 3, 2]]   # Tetrahedron
    try:
        font_attributes = {'title_font_size': 18, 'face_font_size': 15, 'feature_font_size': 10}
        simplicial_visualization = SimplicialVisualization(
            AbstractSimplicialComplex.random(6, edge_set, face_set),
            font_attributes
        )
        simplicial_visualization.show()
    except AssertionError as e:
        logging.error(e)
        self.assertTrue(False)

@unittest.skip('Ignore')
def test_show_3(self):
    edge_set = [[1, 2], [1, 5], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]]
    face_set = [[4, 2, 3], [2, 4, 5]]
    try:
        font_attributes = {'title_font_size': 18, 'face_font_size': 15, 'feature_font_size': 10}
        simplicial_visualization = SimplicialVisualization(
            AbstractSimplicialComplex.random(node_feature_dimension=6, edge_node_indices=edge_set, face_node_indices=face_set),
            font_attributes
        )
        simplicial_visualization.show()
    except AssertionError as e:
        logging.error(e)
        self.assertTrue(False)