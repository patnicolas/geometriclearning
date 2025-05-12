from unittest import TestCase
import itertools


class TestIterTools(TestCase):
    def test_count(self):
        counter_it = itertools.count(start=3, step =4)
        for idx in range(20):
            logging.info(f'{idx} next count: {next(counter_it)}')

    def test_zipper(self):
        data = [idx for idx in range(200, 240)]
        logging.info(f'Length: {len(data)}')
        zipped_data_it = zip(itertools.count(), data)
        for n in range(len(data)):
            logging.info(f'Zipped: {next(zipped_data_it)}')

    def test_repeat(self):
        repeat_it = itertools.repeat(5, times=3)
        logging.info(f'Type counter {type(repeat_it)}')
        cubes = map(pow, range(5), itertools.repeat(3))
        for cube in cubes:
            logging.info(f'Cube: {cube}')
        inv_func = lambda x, n: 1.0/(1.0 + x+ n)
        inverses = map(inv_func, range(10), itertools.repeat(2))
        for inverse in inverses:
            logging.info(f'Inverse: {inverse}')

    def test_chain(self):
        letters = ['a', 'b', 'c', 'd', 'e']
        numbers = [1, 2, 3, 4, 5]
        combines = itertools.chain(letters, numbers)
        for combine in combines:
            logging.info(f'Combine: {combine}')
        # a, b c, d, e, 1, 2, 3, 4, 5

    def test_compress(self):
        letters = ['a', 'b', 'c', 'd', 'e']
        filter = [True, False, False, True,True]
        compressed = itertools.compress(letters, filter)
        for compress in compressed:
            logging.info(f'Compress: {compress}')
        # a, d, 3

    def test_takewhile(self):
        fct = lambda x: pow(x,2) < 32
        results = itertools.takewhile(fct, [n for n in range(30)])
        for result in results:
            logging.info(f'Result: {result}')

    def test_dropwhile(self):
        fct = lambda x: pow(x,2) < 128
        results = itertools.dropwhile(fct, [n for n in range(30)])
        for result in results:
            logging.info(f'_Result: {result}')

    def test_groupby(self):
        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age

            def __str__(self):
                f'Name: {self.name}, age: {self.age}'

        get_age = lambda p: p.age

        persons = [Person("Pat", 23), Person("Emily", 35), Person("Cody", 23)]
        grouped_by = itertools.groupby(persons, get_age)
        for key, group in grouped_by:
            for person in group:
                logging.info(f'KEY: {key} => {person.name}')
