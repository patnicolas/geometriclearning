import unittest
import logging
import os
from dataset.graph.pyg_datasets import PyGDatasets
from topology.simplicial.graph_to_simplicial import GraphToSimplicial, SimplexType
from python import SKIP_REASON


class GraphToSimplicialTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora(self):
        pyg_dataset = PyGDatasets('Cora')
        dataset = pyg_dataset()

        simplicial_generator = GraphToSimplicial(dataset_name='Cora',
                                                 data=dataset[0],
                                                 threshold_clique_complex=21000,
                                                 simplex_types=SimplexType.WithTetrahedrons)
        simplicial_elements = simplicial_generator(num_eigenvectors=[4, 3, 3, 4])
        logging.info(simplicial_elements)

    def test_faces_features(self):
        from dataset.graph.pyg_datasets import PyGDatasets

        pyg_dataset = PyGDatasets('KarateClub')
        dataset = pyg_dataset()
        simplicial_generator = GraphToSimplicial(dataset_name='KarateClub',
                                                 data=dataset[0],
                                                 threshold_clique_complex=21000,
                                                 simplex_types=SimplexType.WithTetrahedrons)
        simplicial_elements = simplicial_generator(num_eigenvectors=[4, 3, 3, 4])
        logging.info(simplicial_elements)


    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_pubmed(self):
        pyg_dataset = PyGDatasets('PubMed')
        dataset = pyg_dataset()

        simplicial_generator = GraphToSimplicial(dataset_name='PubMed',
                                                 data=dataset[0],
                                                 threshold_clique_complex=21000,
                                                 simplex_types=SimplexType.WithTriangles)
        simplicial_complex = simplicial_generator()
        counts = GraphToSimplicial.count_simplex_by_type(simplicial_complex)
        logging.info(counts)
        self.assertEqual(counts['triangles'], 12520)
        self.assertEqual(counts['edges'], 44324)