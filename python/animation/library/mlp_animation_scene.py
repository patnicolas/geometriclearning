

from animation.library.gauge_scatter_plots_scene import GaugeScatterPlotsScene

from manim import *
from animation.library.plots.gauge_vgrp import GaugeVGrp
from animation.library.plots.scatter_2d_vgrp import Scatter2DVGrp
from animation.library.gauge_scatter_plots_scene import GaugeScatterPlotsConfig

class MLPAnimationScene(Scene):
    def construct(self) -> None:
        # Setup Value tracker
        vt = ValueTracker(0)
        # Retrieve the configuration
        composer_config = GaugeScatterPlotsConfig.build()
        # Add Gauge
        gauge_group = self.__add_gauge(vt, composer_config)
        # Add Scatter 2D group located
        scatter_2d_group = self.__add_scatter_2d_group(vt, composer_config, gauge_group)
        self.play(vt.animate.set_value(len(GaugeScatterPlotsScene.data_points) - 1),
                  run_time=composer_config.run_time,
                  rate_func=linear)
        self.wait()

    def __add_network(self, vt: ValueTracker) -> None:

    def __add_gauge(self, vt: ValueTracker, composer_config: GaugeScatterPlotsConfig) -> GaugeVGrp:
        y = [item[1] for item in GaugeScatterPlotsScene.data_points]
        gauge_group = GaugeVGrp(gauge_config=composer_config.gauge_config,
                                data_points=y).to_edge(UR)
        gauge_group.add_updater(gauge_group.get_updater(vt))
        gauge_group.scale(composer_config.display_scale[0])
        self.add(gauge_group)
        return gauge_group

    def __add_scatter_2d_group(self,
                               vt: ValueTracker,
                               composer_config: GaugeScatterPlotsConfig,
                               gauge_group: GaugeVGrp) -> Scatter2DVGrp:
        scatter_2d_group = Scatter2DVGrp(scatter_2d_config=composer_config.scatter_2d_config,
                                         data_points=GaugeScatterPlotsScene.data_points).to_edge(DR)
        scatter_2d_group.add_updater(scatter_2d_group.get_updater(vt))
        scatter_2d_group.scale(composer_config.display_scale[1])
        self.add(scatter_2d_group)
        scatter_2d_group.next_to(gauge_group, DOWN, buff=0.5)
        return scatter_2d_group
