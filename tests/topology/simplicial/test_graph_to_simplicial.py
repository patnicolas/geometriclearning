import unittest
import logging
import os
from dataset.graph.pyg_datasets import PyGDatasets
from topology.simplicial.graph_to_simplicial import GraphToSimplicial, SimpliceTypes, SimplicialFeatures
import toponetx as tnx
from python import SKIP_REASON


class GraphToSimplicialTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora(self):
        pyg_dataset = PyGDatasets('Cora')
        dataset = pyg_dataset()

        simplicial_generator = GraphToSimplicial(dataset_name='Cora',
                                                 data=dataset[0],
                                                 threshold_clique_complex=21000,
                                                 simplice_types=SimpliceTypes.WithTetrahedrons)
        simplicial_complex = simplicial_generator()
        counts = GraphToSimplicial.count_simplices_by_type(simplicial_complex)
        self.assertEqual(counts['triangles'], 1630)
        self.assertEqual(counts['tetrahedrons'], 220)
        logging.info(counts)


    def test_faces_features(self):
        sc = tnx.datasets.karate_club('simplicial')
        simplicial_features = SimplicialFeatures.random(sc, ['node_feat', 'edge_feat', 'face_feat'])
        logging.info(simplicial_features.edge_features())
        logging.info(simplicial_features.show(3))


    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_pubmed(self):
        pyg_dataset = PyGDatasets('PubMed')
        dataset = pyg_dataset()

        simplicial_generator = GraphToSimplicial(dataset_name='PubMed',
                                                 data=dataset[0],
                                                 threshold_clique_complex=21000,
                                                 simplice_types=SimpliceTypes.WithTriangles)
        simplicial_complex = simplicial_generator()
        counts = GraphToSimplicial.count_simplices_by_type(simplicial_complex)
        logging.info(counts)
        self.assertEqual(counts['triangles'], 12520)
        self.assertEqual(counts['edges'], 44324)