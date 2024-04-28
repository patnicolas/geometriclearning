from unittest import TestCase
import unittest
import pandas as pd
import numpy as np


class TestVectorizer(TestCase):
    @unittest.skip("Not needed")
    def test_dict_vectorizer(self):
        from sklearn.feature_extraction import DictVectorizer

        token_dict =[
            {'hello':1,'patrick':1,'this':1,'is':1,'not':1,'a':2,'or':1,'joke':1},
            {'the':1,'joke':1,'is':1,'on':1,'you':1}
        ]
        dv = DictVectorizer()
        dv.fit(token_dict)
        print(dv.vocabulary_)
        x = dv.transform(token_dict)
        print(x)
        df = pd.DataFrame(x, columns=['a', 'hello', 'is', 'joke', 'not', 'on', 'or', 'patrick', 'the', 'this', 'you'])
        print(df)

    def test_generator_exp(TestCase):
        values = np.random.rand(8,10)
        # Option 1: Direct iterator
        it1 = (10 + y for x in values for y in x)
        while True:
            try:
                next_val = next(it1)
                print(next_val)
            except StopIteration as e:
                print('Completed!')
                break

        # Option 2: Generator expression
        def add_gen(input: np.array) -> np.array:
            for x in input:
                for y in x:
                    yield 10 + y

        prod = add_gen(values)  # Invoke the generator
        it = iter(prod)         # Convert to an iterator
        while True:
            try:
                next_value = next(it)
                print(next_value)
            except StopIteration as e:
                print('completed')
                break

