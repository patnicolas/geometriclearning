import unittest
import logging
import python
from topology.simplicial.abstract_simplicial_complex_builder import AbstractSimplicialComplexBuilder


class AbstractSimplicialComplexBuilderTest(unittest.TestCase):

    def test_init_1(self):
        dataset = 'Cora'
        simplicial_complex_builder = AbstractSimplicialComplexBuilder.build(dataset)
        logging.info(simplicial_complex_builder)