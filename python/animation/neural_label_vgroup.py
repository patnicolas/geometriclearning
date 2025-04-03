__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *
from typing import List
from neural_config import NeuralConfig


class NeuralLabelVGroup(VGroup):
    def __init__(self,
                 layers: List[VMobject],
                 layer_sizes: List[int],
                 *args,
                 **kwargs) -> None:
        VGroup.__init__(self, *args, **kwargs)
        self.labels = NeuralLabelVGroup.__add_labels(layers, layer_sizes)

    """ ------------------------------  Private Helper Methods ---------------  """

    @staticmethod
    def __add_labels(layers:  List[VMobject], layer_sizes: List[int]) -> VGroup:
        return VGroup(
            *[NeuralLabelVGroup.__label_inputs(layers, layer_sizes), NeuralLabelVGroup.__label_outputs(layers)]
        )

    @staticmethod
    def __label_inputs(layers:  List[VMobject], layer_sizes: List[int]) -> VGroup:
        input_labels = VGroup()
        overflow = layer_sizes[0] > NeuralConfig['max_shown_neurons']
        half_display_point = NeuralConfig['max_shown_neurons'] // 2
        for n, neuron in enumerate(layers[0].neurons):
            index = n + 1
            if overflow and n >= half_display_point:
                index += layer_sizes[0] - NeuralConfig['max_shown_neurons']

            label = MathTex(rf"{{x}}_{{{index}}}", font_size=36)
            label.next_to(neuron, LEFT, buff=0.2)
            input_labels.add(label)
        return input_labels

    @staticmethod
    def __label_outputs(layers:  List[VMobject]):
        output_labels = VGroup()
        for n, neuron in enumerate(layers[-1].neurons):
            index = n + 1
            label = MathTex(rf"\hat{{y}}_{{{index}}}", font_size=42)
            label.next_to(neuron, RIGHT, buff=0.2)
            output_labels.add(label)
        return output_labels

    @staticmethod
    def __label_outputs_text(layers:  List[VMobject], outputs) -> VGroup:
        output_text = VGroup()
        for n, neuron in enumerate(layers[-1].neurons):
            label = Text(outputs[n])
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            label.shift((neuron.get_width() + label.get_width()) * RIGHT)
            output_text.add(label)
        return output_text

