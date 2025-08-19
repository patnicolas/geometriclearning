import unittest
import logging
import os
from dataset.graph.pyg_datasets import PyGDatasets
from topology.graph_to_simplicial import GraphToSimplicial, SimpliceTypes
import python
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

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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