import unittest
from fractal.fractaldimimage import FractalDimImage


class FractalDimImageTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        image_path_name = '../../../images/fractal_test_image.jpg'
        fractal_dim_image = FractalDimImage(image_path_name)
        self.assertTrue(fractal_dim_image.image is not None)
        if fractal_dim_image.image is not None:
            print(fractal_dim_image.image.shape)

    @unittest.skip('Ignore')
    def test_call(self):
        import numpy as np
        image_path_name = '../../../images/fractal_test_image.jpg'

        fractal_dim_image = FractalDimImage(image_path_name)
        self.assertTrue(fractal_dim_image.image is not None)
        fractal_dim, trace = fractal_dim_image()
        trace_str = '/n'.join([str(box_param) for box_param in trace])
        print(f'Fractal dimension: {float(fractal_dim)}\nTrace {trace_str}')

    @unittest.skip('Ignore')
    def test_timeit(self):
        import timeit
        image_path_name = '../../../images/fractal_test_image.jpg'
        timeit.timeit(FractalDimImage(image_path_name))
        image_path_name = '../../../images/fractal_test_image_large.jpg'
        timeit.timeit(FractalDimImage(image_path_name))


    @unittest.skip('Ignore')
    def test_plots(self):
        import numpy as np
        image_path_name = '../../../images/fractal_test_image.jpg'
        fractal_dim_image = FractalDimImage(image_path_name)
        self.assertTrue(fractal_dim_image.image is not None)
        fractal_dim, trace = fractal_dim_image()
        trace_str = '/n'.join([str(box_param) for box_param in trace])
        print(f'Fractal dimension: {float(fractal_dim)}\nTrace {trace_str}')

        box_params = np.array([[param.eps, param.measurements] for param in trace])
        y = box_params[:, 1]
        x = np.linspace(1, len(y), len(y))
        import matplotlib.pyplot as plt

        # Create a scatter plot
        plt.scatter(x, y)

        # Add title and labels
        plt.title('Trace box measurement distribution - image')
        plt.xlabel('Iterations')
        plt.ylabel('Box measurement units')

        plt.show()

    def test_pandas(self):
        import pandas as pd
        obj = pd.Series([4, 9, 12])
        print(obj.array)
        obj = pd.Series([4, 12, 55, 19, 1], index=['l1', 'l2', 'l3', 'l4', 'l5'])
        print(f'Index: {obj.index} Array: {obj.array}')
        print(f"obj[l2]:{obj[['l2', 'l4']]}")
        obj['l2'] = 0
        print(f"obj[l2]:{obj['l2']}")
        # Operation
        print(f'Filter obj[obj< 20]: {obj[obj < 10]}')
        print(f'Filter obj*3: {obj*3}')
        # Test if a label or index belongs to a series
        is_valid = 'l3' in obj

        # Convert a dictionary to a PD Series and back
        converted_pd = pd.Series({'a': 9, 'b': -3, 'c': 9})
        print(converted_pd)
        print(converted_pd.to_dict())
        obj2 = pd.Series(converted_pd, index=['a', 'b', 'c', 'd'])
        print(obj2)
        print(f"Value for index=d:\n {pd.isna(obj2)}")    #a    False b    False c    False  d     True


