from unittest import TestCase
import unittest
import torch
from util.imagetensor import ImageTensor
img_dir = '../../output/test1'


class TestImageTensor(TestCase):
    def test_img(self):
        from PIL import Image
        import matplotlib.pyplot as plt
        import numpy as np

        values = [[[255, 0, 0],[255,0,0]],[[255, 0, 0],[255,0,0]]]
        np_values = np.array(values)
        print(np_values.shape)
        img = Image.fromarray(np_values, 'RGB')
        plt.imshow(img)
        img.save('../../images/test6.png')
        plt.show()

    @unittest.skip("NO reason")
    def test_to_image(self):
        try:
            from PIL import Image
            import numpy
            im = Image.open("../../images/t.png")
            np_im = numpy.array(im)
            print(np_im.shape)
            np_im = np_im - 23
            new_im = Image.fromarray(np_im)
            new_im.save("../../output/altered_t.png")
            """
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.imshow(data, cmap='rgb', interpolation='none')
            fig.show()
            """
        except Exception as e:
            self.fail(str(e))

    @unittest.skip("NO reason")
    def test_to_images(self):
        try:
            h = 256
            w = 480
            input_tensor1 = torch.rand((h, w), dtype=torch.float)
            input_tensor2 = torch.full((h, w), 0.5)
            image_conversion = ImageTensor(img_dir)
            image_conversion.to_images([input_tensor1, input_tensor2])
        except Exception as e:
            print(str(e))
            self.fail()

    @unittest.skip("NO reason")
    def test_to_dataset(self):
        try:
            image_conversion = ImageTensor(img_dir)
            data_loader = image_conversion.to_dataset()
            it = iter(data_loader)
            print(it.next())
        except Exception as e:
            print(str(e))
            self.fail()