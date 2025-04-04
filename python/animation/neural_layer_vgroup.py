__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *
from typing import AnyStr, Any, Dict, List
from neural_config import NeuralConfig


class NeuralLayerVGroup(VGroup):
    def __init__(self,
                 layer_sizes: List[int],
                 *args,
                 **kwargs) -> None:
        VGroup.__init__(self, *args, **kwargs)
        self.layers = NeuralLayerVGroup.__add_neurons(layer_sizes)

    """ ------------------------------  Private Helper Methods ---------------  """

    @staticmethod
    def __add_neurons(layer_sizes: List[int]) -> VGroup:
        layers = VGroup(*[
            NeuralLayerVGroup.__get_layer(layer_sizes, size, index)
            for index, size in enumerate(layer_sizes)
        ])
        layers.arrange_submobjects(RIGHT, buff=NeuralConfig['layer_to_layer_buff'])
        return layers

    @staticmethod
    def __get_layer(layer_sizes: List[int], size: int, index: int) -> VGroup:
        layer = VGroup()
        n_neurons = size
        if n_neurons > NeuralConfig['max_shown_neurons']:
            n_neurons = NeuralConfig['max_shown_neurons']

        neurons = VGroup(*[
            Circle(
                radius=NeuralConfig['neuron_radius'],
                stroke_color=NeuralLayerVGroup.__get_nn_fill_color(layer_sizes, index),
                stroke_width=NeuralConfig['neuron_stroke_width'],
                fill_color=GREY,
                fill_opacity=NeuralConfig['neuron_fill_opacity'],
            )
            for _ in range(n_neurons)
        ])
        neurons.arrange_submobjects(
            DOWN, buff=NeuralConfig['neuron_to_neuron_buff']
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        text = MathTex(rf"{{{layer_sizes[index]}}} \ units", font_size=32)
        text.next_to(neurons[-1], DOWN, buff=0.5)
        layer.add(text)

        if size > n_neurons:
            dots = Tex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if NeuralConfig['brace_for_large_layers']:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)
        return layer

    @staticmethod
    def __get_nn_fill_color(layer_sizes: List[int], index: int,) -> VMobject:
        if index == -1 or index == len(layer_sizes) - 1:
            return NeuralConfig['output_neuron_color']
        return NeuralConfig['input_neuron_color'] if index == 0 else NeuralConfig['hidden_layer_neuron_color']
