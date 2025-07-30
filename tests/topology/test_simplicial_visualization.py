import unittest
import logging
import python
from topology.simplicial_feature_set import SimplicialFeatureSet
from topology.simplicial_visualization import SimplicialVisualization


class SimplicialVisualizationTest(unittest.TestCase):

    def test_show_1(self):
        edge_set = [[1, 2], [1, 3], [2, 3], [5, 6], [3, 4], [1, 6], [4, 5], [3, 6], [4, 6], [6, 7], [7, 1],
                    [1, 8], [3, 8], [5, 8], [6, 9], [7, 9], [7, 10], [8, 10], [9, 10], [4, 11], [5, 11], [6, 11],
                    [3, 12], [8, 12], [9, 13], [3, 13], [2, 14], [10, 15], [11, 15], [14, 16], [1, 17], [11, 18],
                    [9, 19], [9, 20], [20, 21]]
        face_set = [[4, 3, 6], [1, 2, 3], [5, 4, 6], [6, 7, 11],  [6, 7, 9], [4, 5, 10], [7, 8, 10], [4, 6, 9],
                    [3, 7, 12], [12, 14, 16], [2, 15, 16], [3, 13, 14], [9, 11, 13], [4, 17, 18], [6, 16, 17],
                    [1, 18, 19], [8, 13, 14], [14, 21, 19],
                    [3, 4, 5, 8], [2, 1, 7, 6], [1, 2, 3, 10], [14, 15, 16, 17]]
        try:
            logging.info(f'Number of edges: {len(edge_set)}')
            logging.info(f'Number of faces: {len(face_set)}')
            font_attributes = {'title_font_size': 18, 'face_font_size': 15, 'feature_font_size': 10 }
            simplicial_visualization = SimplicialVisualization(
                SimplicialFeatureSet.build(3, edge_set, face_set),
                font_attributes
            )
            simplicial_visualization.show()
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_show_2(self):
        edge_set = [
            [1, 2], [1, 3], [2, 3], [5, 6], [3, 4], [1, 6], [4, 5], [3, 6], [4, 6], [6, 7], [7, 1], [2, 6]
        ]
        face_set = [
            [4, 3, 6], [1, 2, 3], [5, 4, 6], [6, 7, 1],  # Triangle
            [6, 4, 3, 2]]   # Tetrahedron
        try:
            font_attributes = {'title_font_size': 18, 'face_font_size': 15, 'feature_font_size': 10}
            simplicial_visualization = SimplicialVisualization(
                SimplicialFeatureSet.build(6, edge_set, face_set),
                font_attributes
            )
            simplicial_visualization.show()
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)