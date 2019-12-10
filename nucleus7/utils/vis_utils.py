# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Tools for visualization of model DNA
"""

from collections import OrderedDict
from collections import namedtuple
from functools import partial
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import warnings

from matplotlib import axes as plt_axes
from matplotlib import patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from nucleus7.coordinator.callback import CoordinatorCallback
from nucleus7.coordinator.predictors import PredictorNucleotide
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.data.data_feeder import DataFeeder
from nucleus7.data.dataset import Dataset
from nucleus7.data.processor import DataProcessor
from nucleus7.data.reader import DataReader
from nucleus7.kpi.accumulator import KPIAccumulator
from nucleus7.kpi.kpi_callback import KPIEvaluatorCallback
from nucleus7.kpi.kpi_plugin import KPIPlugin
from nucleus7.model.loss import ModelLoss
from nucleus7.model.metric import ModelMetric
from nucleus7.model.plugin import ModelPlugin
from nucleus7.model.postprocessor import ModelPostProcessor
from nucleus7.model.summary import ModelSummary
from nucleus7.utils import nucleotide_utils

# pylint: disable=invalid-name
# these are type constants, not a class
_NUCLEOTIDE_PLOT = namedtuple("NUCLEOTIDE_PLOT",
                              ["body", "incoming_keys", "generated_keys"])


# pylint: enable=invalid-name


def draw_dna_helix(dna_helix_graph: nx.DiGraph,
                   *, title_prefix: str = "", title_suffix: str = "",
                   figsize=(14, 14), verbosity: int = 0
                   ) -> plt_axes.Subplot:
    """
    Draw interactive dna helix

    All nucleotides bodies are clickable; also the font is automatic rescalable
    to fit it to the circles / wedges.

    By single click on nucleotide, the nucleotide description pops up on the
    right side of image.

    By double click on nucleotide, new figure is opened and subgraph with all
    connections of this nucleotide is drawn in same fashion with verbosity=2.

    Parameters
    ----------
    dna_helix_graph
        directed graph with nucleotides as nodes
    title_prefix
        prefix that will be added to the plot title
    title_suffix
        suffix that will be added to the plot title
    figsize
        size of figure
    verbosity
        verbosity of the visualization;
        if verbosity == 0, then only the connections between
        nucleotide are drawn, otherwise connections between nucleotide keys
        are drawn

    Returns
    -------
    subplot
        axes subplot with drawn dna helix

    Raises
    ------
    ValueError
        if verbosity not in [0, 1]
    """
    if verbosity not in [0, 1]:
        raise ValueError("verbosity can be 0 or 1!")

    figure, subplot = _create_figure_with_subplot(figsize)
    figure.suptitle(title_prefix + "DNA helix" + title_suffix)
    draw_dna_helix_on_subplot(dna_helix_graph, subplot, verbosity=verbosity)
    _draw_click_instructions(subplot)
    _add_subgraph_events(figure, dna_helix_graph)
    figure.canvas.blit(subplot.bbox)
    return subplot


def draw_dna_helix_on_subplot(dna_helix_graph, subplot,
                              radius: int = 10, verbosity: int = 0):
    """
    Draw the dna helix on given subplot according to verbosity

    Parameters
    ----------
    dna_helix_graph
        directed graph with nucleotides as nodes
    subplot
        axes subplot to draw on
    radius
        radius of nucleotide to draw
    verbosity
        verbosity of the visualization;
        if verbosity == 0, then only the connections between
        nucleotide are drawn, otherwise connections between nucleotide keys
        are drawn
    """
    nucleotide_positions = _get_nucleotide_positions(
        dna_helix_graph, radius)
    nucleotide_plots = {}
    for each_nucleotide, each_nucleotide_center in (
            nucleotide_positions.items()):
        nucleotide_plot = draw_nucleotide(
            each_nucleotide, each_nucleotide_center, subplot,
            radius=radius)
        nucleotide_plots[each_nucleotide] = nucleotide_plot
    draw_dna_connections(dna_helix_graph,
                         nucleotide_plots,
                         subplot=subplot, verbosity=verbosity)
    subplot.add_callback(lambda subplot_: subplot_.set_aspect('equal'))
    subplot.autoscale_view()
    subplot.axis('off')
    _draw_legend(subplot)
    _add_update_events(subplot, dna_helix_graph, nucleotide_plots)
    subplot.pchanged()
    subplot.figure.canvas.draw()
    subplot.figure.canvas.flush_events()


def draw_nucleotide(nucleotide: Nucleotide,
                    center: Optional[Sequence[float]] = (0, 0),
                    subplot: Optional[plt_axes.Subplot] = None,
                    radius: float = 10.0) -> _NUCLEOTIDE_PLOT:
    """
    Draw nucleotide with its keys

    Parameters
    ----------
    nucleotide
        directed graph with nucleotides as nodes
    center
        center of nucleotide on the plot
    subplot
        subplot to use
    radius
        radius of nucleotide to draw

    Returns
    -------
    nucleotide_plot
        named tuple holding all the drawn patches for nucleotide body and
        it's keys
    """

    def _callback(subplot_):
        for each_patch in subplot_.patches:
            each_patch.pchanged()

    if subplot is None:
        _, subplot = _create_figure_with_subplot()
    nucleotide_body = _draw_nucleotide_body(
        nucleotide, center, subplot, radius=radius)
    incoming_keys_patches = _draw_nucleotide_incoming_keys(
        nucleotide, center, subplot=subplot,
        nucleotide_body_patch=nucleotide_body,
        radius=radius, width=radius / 2)
    generated_keys_patches = _draw_nucleotide_generated_keys(
        nucleotide, center, subplot=subplot,
        nucleotide_body_patch=nucleotide_body,
        radius=radius, width=radius / 2)

    subplot.add_callback(_callback)
    result = _NUCLEOTIDE_PLOT(body=nucleotide_body,
                              incoming_keys=incoming_keys_patches,
                              generated_keys=generated_keys_patches)
    return result


def draw_dna_connections(dna_helix_graph: nx.DiGraph,
                         nucleotide_plots: Dict[Nucleotide, _NUCLEOTIDE_PLOT],
                         subplot: plt_axes.Subplot,
                         verbosity: int = 0):
    """
    Draw dna connections on given subplot according to verbosity level

    Parameters
    ----------
    dna_helix_graph
        directed graph with nucleotides as nodes
    nucleotide_plots
        mapping of nucleotide to its plot
    subplot
        subplot to draw on
    verbosity
        verbosity of the visualization;
        if verbosity == 0, then only the connections between
        nucleotide are drawn, otherwise connections between nucleotide keys
        are drawn
    """
    nucleotide_positions = {
        each_nucleotide: each_nucleotide_plot.body.center
        for each_nucleotide, each_nucleotide_plot in nucleotide_plots.items()}
    _draw_dna_connections(
        subplot, dna_helix_graph, nucleotide_positions, nucleotide_plots,
        verbosity=verbosity)
    subplot.add_callback(
        partial(_draw_dna_connections,
                dna_helix_graph=dna_helix_graph,
                nucleotide_positions=nucleotide_positions,
                nucleotide_plots=nucleotide_plots, verbosity=verbosity))


def _create_figure_with_subplot(figsize=None
                                ) -> Tuple[plt.Figure, plt_axes.Subplot]:
    figure = plt.figure(figsize=figsize)
    subplot = figure.add_subplot(111)
    figure.show()
    figure.canvas.draw()
    return figure, subplot


def _get_nucleotide_positions(dna_helix_graph: nx.DiGraph, radius: int = 10
                              ) -> Dict[Nucleotide, Sequence[float]]:
    nucleotide_min_distance = 5 * radius
    try:
        nucleotide_positions = nx.nx_agraph.graphviz_layout(
            dna_helix_graph, prog="dot", args="-Goverlap=compress -Granksep=5")
    except ImportError:
        msg = ("For better layout of dna visualization install pygraphviz "
               "(https://pygraphviz.github.io/)")
        warnings.warn(msg, ImportWarning)
        nucleotide_positions = nx.spring_layout(dna_helix_graph)
    edge_lengths = {}
    for each_edge in dna_helix_graph.edges:
        coord0 = np.array(nucleotide_positions[each_edge[0]])
        coord1 = np.array(nucleotide_positions[each_edge[1]])
        edge_length = np.sqrt(np.sum((coord1 - coord0) ** 2))
        edge_lengths[each_edge] = edge_length
    if edge_lengths:
        min_edge_length = min(edge_lengths.values())
    else:
        if nucleotide_positions:
            min_coord = np.min(list(nucleotide_positions.values()), 0)
            max_coord = np.max(list(nucleotide_positions.values()), 0)
            min_edge_length = (np.sqrt(np.sum((max_coord - min_coord) ** 2))
                               / len(nucleotide_positions))
        else:
            min_edge_length = 1
    coord_scale_factor = nucleotide_min_distance / min_edge_length
    nucleotide_positions_scaled = {
        each_node: [coord * coord_scale_factor for coord in each_position]
        for each_node, each_position in nucleotide_positions.items()}
    return nucleotide_positions_scaled


def _add_update_events(subplot: plt_axes.Subplot, dna_helix_graph: nx.DiGraph,
                       nucleotide_plots: Dict[Nucleotide, _NUCLEOTIDE_PLOT]):
    subplot.figure.canvas.mpl_connect(
        'draw_event', lambda x: subplot.pchanged())
    subplot.figure.canvas.mpl_connect(
        'resize_event', lambda x: subplot.pchanged())

    text_initial_position = list(nucleotide_plots.values())[0].body.center
    text_object = subplot.text(
        text_initial_position[0], text_initial_position[1], "",
        ha="right", va="top", ma="left",
        bbox=dict(facecolor='white', edgecolor='blue', pad=5.0))
    text_object.set_visible(False)

    subplot.figure.canvas.mpl_connect(
        'button_press_event',
        partial(_remove_nucleotide_info_text, text_object=text_object))
    subplot.figure.canvas.mpl_connect(
        'pick_event',
        partial(_draw_nucleotide_info, dna_helix_graph=dna_helix_graph,
                text_object=text_object, subplot=subplot))


def _add_subgraph_events(figure: plt.Figure, dna_helix_graph: nx.DiGraph):
    figure.canvas.mpl_connect(
        'pick_event',
        partial(_create_subgraph_plot, dna_helix_graph=dna_helix_graph))


def _remove_nucleotide_info_text(event, text_object: plt.Text):
    # pylint: disable=unused-argument
    # event argument is there for interface even if it is not used here
    text_object.set_text("")
    text_object.set_visible(False)
    plt.show()


def _draw_nucleotide_info(event, dna_helix_graph: nx.DiGraph,
                          text_object: plt.Text, subplot: plt_axes.Subplot):
    mouseevent = event.mouseevent
    if mouseevent.dblclick or mouseevent.button != 1:
        return

    nucleotide_name = event.artist.get_label().split(":")[-1]
    nucleotide = _get_nucleotide_by_name(nucleotide_name, dna_helix_graph)
    nucleotide_info = nucleotide_utils.get_nucleotide_info(
        nucleotide=nucleotide)

    nucleotide_info_text = "Information for {}:\n\n".format(
        nucleotide_name).upper()
    nucleotide_info_text += nucleotide_utils.format_nucleotide_info(
        nucleotide_info)

    xlim = subplot.get_xlim()
    ylim = subplot.get_ylim()
    text_coord = (xlim[1] - 0.05 * (xlim[1] - xlim[0]),
                  ylim[1] - 0.05 * (ylim[1] - ylim[0]))

    text_object.set_text(nucleotide_info_text)
    text_object.set_x(text_coord[0])
    text_object.set_y(text_coord[1])
    text_object.set_visible(True)
    plt.show()


def _create_subgraph_plot(event, dna_helix_graph: nx.DiGraph):
    mouseevent = event.mouseevent
    if not mouseevent.dblclick or mouseevent.button != 1:
        return

    logger = logging.getLogger(__name__)
    nucleotide_name = event.artist.get_label().split(":")[-1]
    nucleotide = _get_nucleotide_by_name(nucleotide_name, dna_helix_graph)
    logger.info("Create subgraph plot for %s", nucleotide_name)
    figure, subplot = _create_figure_with_subplot()
    figure.suptitle("Subgraph of nucleotide {}".format(nucleotide_name))

    nucleotide_with_neighbors_subgraph = _get_nucleotide_subgraph(
        dna_helix_graph, nucleotide)
    draw_dna_helix_on_subplot(
        nucleotide_with_neighbors_subgraph, subplot, verbosity=1)
    _draw_click_instructions(subplot, doubleclick=False)
    plt.draw()
    logger.info("Done!")


def _get_nucleotide_by_name(nucleotide_name: str, dna_helix_graph: nx.DiGraph
                            ) -> Optional[Nucleotide]:
    nucleotide = None
    for each_nucleotide in dna_helix_graph:
        if each_nucleotide.name == nucleotide_name:
            nucleotide = each_nucleotide
            break
    return nucleotide


def _draw_legend(subplot: plt_axes.Subplot):
    nucleotide_patches, nucleotide_class_names_with_names = (
        subplot.figure.gca().get_legend_handles_labels())
    nucleotide_class_names = [
        each_class_name_with_name.split(":")[0]
        for each_class_name_with_name in nucleotide_class_names_with_names]
    legend_labels, legend_items = zip(*OrderedDict(
        zip(nucleotide_class_names, nucleotide_patches)).items())
    subplot.legend(legend_items, legend_labels, loc="lower right",
                   bbox_to_anchor=(0, 0), title="Nucleotide types")


def _draw_dna_connections(
        subplot: plt_axes.Subplot,
        dna_helix_graph: nx.DiGraph,
        nucleotide_positions: Dict[Nucleotide, tuple],
        nucleotide_plots: Dict[Nucleotide, _NUCLEOTIDE_PLOT],
        verbosity: int = 0):
    # assumes that the coordinates are equal scaled in the view
    edge_label = "_dna_edge"
    _remove_dna_edge_patches(subplot, edge_label)

    nucleotide_body_patch = list(nucleotide_plots.values())[0].body
    body_patch_window_extent = nucleotide_body_patch.get_window_extent()
    if verbosity == 0:
        node_size_pixels = (body_patch_window_extent.width / 2 * 1.5
                            * subplot.figure.dpi)
        edge_patches = _draw_without_key_connections(
            dna_helix_graph, node_size_pixels, nucleotide_positions, subplot)
        if edge_patches:
            for each_edge_patch in edge_patches:
                each_edge_patch.set_zorder(0)
    else:
        edge_patches = _draw_with_key_connections(
            dna_helix_graph, nucleotide_plots, subplot)

    edge_patches = edge_patches or []
    for each_edge_patch in edge_patches:
        each_edge_patch.set_label(edge_label)


def _draw_without_key_connections(dna_helix_graph, node_size_pixels,
                                  nucleotide_positions, subplot
                                  ) -> List[patches.FancyArrowPatch]:
    edge_patches = nx.draw_networkx_edges(
        dna_helix_graph, pos=nucleotide_positions,
        node_size=node_size_pixels, ax=subplot)
    return edge_patches


def _draw_with_key_connections(dna_helix_graph, nucleotide_plots, subplot
                               ) -> List[patches.FancyArrowPatch]:
    edge_patches = []
    for each_nucleotide in dna_helix_graph:
        edge_patches_for_nucleotide = _draw_nucleotide_keys_connections(
            each_nucleotide, dna_helix_graph, nucleotide_plots)
        if edge_patches_for_nucleotide:
            edge_patches.extend(edge_patches_for_nucleotide)
    for each_edge_patch in edge_patches:
        subplot.add_patch(each_edge_patch)
    return edge_patches


def _draw_nucleotide_keys_connections(nucleotide, dna_helix_graph,
                                      nucleotide_plots
                                      ) -> List[patches.FancyArrowPatch]:
    edge_patches_for_nucleotide = []
    predecessors = dna_helix_graph.predecessors(nucleotide)
    successors = dna_helix_graph.successors(nucleotide)
    for each_inbound_nucleotide in predecessors:
        edge_patches = _draw_key_connections_for_nucleotide_pair(
            nucleotide, each_inbound_nucleotide, nucleotide_plots)
        edge_patches_for_nucleotide.extend(edge_patches)
    for each_outgoing_nucleotide in successors:
        edge_patches = _draw_key_connections_for_nucleotide_pair(
            each_outgoing_nucleotide, nucleotide, nucleotide_plots)
        edge_patches_for_nucleotide.extend(edge_patches)
    return edge_patches_for_nucleotide


def _draw_key_connections_for_nucleotide_pair(
        nucleotide: Nucleotide,
        inbound_nucleotide: Nucleotide,
        nucleotide_plots: Dict[Nucleotide, _NUCLEOTIDE_PLOT]
) -> List[patches.FancyArrowPatch]:
    nucleotide_plot = nucleotide_plots[nucleotide]
    incoming_nucleotide_plot = nucleotide_plots[inbound_nucleotide]

    inputs_to_nucleotide = {each_inbound_node: {"": None}
                            for each_inbound_node in nucleotide.inbound_nodes}
    inputs_to_nucleotide[inbound_nucleotide.name] = {
        k: k for k in inbound_nucleotide.generated_keys_all}

    inputs_to_nucleotide_filtered = nucleotide.filter_inputs(
        inputs_to_nucleotide)
    if "" in inputs_to_nucleotide_filtered:
        del inputs_to_nucleotide_filtered[""]

    edge_patches = []
    for each_incoming_key, each_generated_key in (
            inputs_to_nucleotide_filtered.items()):
        edge_patch = _draw_edge_between_keys(
            nucleotide, inbound_nucleotide,
            each_incoming_key, each_generated_key,
            nucleotide_plot, incoming_nucleotide_plot)
        edge_patches.append(edge_patch)
    return edge_patches


def _draw_edge_between_keys(nucleotide: Nucleotide,
                            inbound_nucleotide: Nucleotide,
                            incoming_key: str,
                            generated_key: str,
                            nucleotide_plot: _NUCLEOTIDE_PLOT,
                            inbound_nucleotide_plot: _NUCLEOTIDE_PLOT
                            ) -> patches.FancyArrowPatch:
    if incoming_key not in nucleotide.incoming_keys_all:
        incoming_key = "DYNAMIC"
    if generated_key not in inbound_nucleotide.generated_keys_all:
        generated_key = "DYNAMIC"
    generated_key_wedge_patch = (
        inbound_nucleotide_plot.generated_keys[
            generated_key])
    incoming_key_wedge_patch = (
        nucleotide_plot.incoming_keys[incoming_key])
    position0, _ = _get_wedge_center_and_angle(
        generated_key_wedge_patch.center,
        generated_key_wedge_patch.r - generated_key_wedge_patch.width,
        generated_key_wedge_patch.theta1,
        generated_key_wedge_patch.theta2,
        generated_key_wedge_patch.width * 2,
    )
    position1, _ = _get_wedge_center_and_angle(
        incoming_key_wedge_patch.center,
        incoming_key_wedge_patch.r - incoming_key_wedge_patch.width,
        incoming_key_wedge_patch.theta1,
        incoming_key_wedge_patch.theta2,
        incoming_key_wedge_patch.width * 2,
    )
    seed = np.abs(
        hash(":".join([inbound_nucleotide.name, incoming_key,
                       nucleotide.name,
                       generated_key])) >> 32)
    seed = np.clip(seed, 1, 2 ** 32 - 2)
    np.random.seed(seed)
    color = np.random.rand(3)
    edge_patch = patches.FancyArrowPatch(
        posA=position0, posB=position1,
        arrowstyle='-|>',
        mutation_scale=10,
        edgecolor=color,
    )
    return edge_patch


def _get_nucleotide_subgraph(dna_helix_graph: nx.DiGraph, nucleotide):
    predecessors = list(dna_helix_graph.predecessors(nucleotide))
    successors = list(dna_helix_graph.successors(nucleotide))
    neighbors_with_nucleotide = predecessors + [nucleotide] + successors
    return nx.subgraph(dna_helix_graph, neighbors_with_nucleotide)


def _remove_dna_edge_patches(subplot, edge_label):
    for each_patch in subplot.patches[:]:
        if each_patch.get_label() == edge_label:
            each_patch.remove()


def _draw_nucleotide_body(nucleotide, center, subplot: plt_axes.Subplot,
                          radius=10.0):
    nucleotide_color, nucleotide_base_class = _get_nucleotide_color(nucleotide)
    nucleotide_name = nucleotide.name
    if len(nucleotide_name) > 10:
        nucleotide_name = nucleotide_name.replace("_", "_\n")

    nucleotide_body = patches.Circle(
        center, radius=radius, color=nucleotide_color)
    text_object = subplot.text(
        center[0], center[1], nucleotide_name, va="center", ha="center")
    text_object.draw(subplot.figure.canvas.renderer)
    subplot.add_patch(nucleotide_body)
    nucleotide_body.add_callback(
        partial(_nucleotide_name_callback, text_object=text_object))
    nucleotide_body.set_label(":".join([nucleotide_base_class.__name__,
                                        nucleotide.name]))
    nucleotide_body.set_picker(True)
    return nucleotide_body


def _draw_nucleotide_incoming_keys(nucleotide, center,
                                   nucleotide_body_patch: patches.Circle,
                                   subplot: plt_axes.Subplot,
                                   radius=10.0, width=5.0):
    theta_min, theta_max = 0, 180
    keys_color = _get_key_color("incoming")
    incoming_keys_required = nucleotide.incoming_keys_required
    if nucleotide.dynamic_incoming_keys:
        incoming_keys_required = incoming_keys_required + ["DYNAMIC"]
    key_patches = _draw_nucleotide_keys(
        incoming_keys_required, nucleotide.incoming_keys_optional,
        center, nucleotide_body_patch=nucleotide_body_patch,
        subplot=subplot, color=keys_color, radius=radius, width=width,
        theta_min=theta_min, theta_max=theta_max)
    return key_patches


def _draw_nucleotide_generated_keys(nucleotide, center,
                                    nucleotide_body_patch: patches.Circle,
                                    subplot: plt_axes.Subplot,
                                    radius=10.0, width=5.0):
    theta_min, theta_max = 180, 360
    keys_color = _get_key_color("generated")
    generated_keys_required = nucleotide.generated_keys_required
    if nucleotide.dynamic_generated_keys:
        generated_keys_required = generated_keys_required + ["DYNAMIC"]
    key_patches = _draw_nucleotide_keys(
        generated_keys_required, nucleotide.generated_keys_optional,
        center, nucleotide_body_patch=nucleotide_body_patch,
        subplot=subplot, color=keys_color, radius=radius, width=width,
        theta_min=theta_min, theta_max=theta_max)
    return key_patches


def _draw_nucleotide_keys(keys_required: Optional[List[str]],
                          keys_optional: Optional[List[str]],
                          center: Sequence[float],
                          nucleotide_body_patch: patches.Circle,
                          subplot: plt_axes.Subplot,
                          color, radius=10.0, width=5.0,
                          theta_min=0, theta_max=180):
    # pylint: disable=too-many-arguments
    # not possible to have less arguments without more complexity
    # pylint: disable=too-many-locals
    # is not possible to split the method without more complexity
    keys_required = keys_required or []
    keys_optional = keys_optional or []

    keys_all = keys_required + keys_optional

    if not keys_all:
        return None

    number_of_keys = len(keys_all)
    theta_delta = (theta_max - theta_min) / number_of_keys
    thetas = np.arange(theta_min, theta_max, theta_delta)
    key_patches = {}
    for each_key, each_start_theta in zip(keys_all, thetas):
        hatch = "x" if each_key in keys_optional else None
        theta1 = each_start_theta
        theta2 = theta1 + theta_delta
        key_patch = patches.Wedge(
            center, radius + width, theta1, theta2, width=width,
            facecolor=color, hatch=hatch, edgecolor="white")
        key_patches[each_key] = key_patch

        text_object = _draw_key_text(
            each_key, center, theta1, theta2, radius, width, subplot)
        key_patch.add_callback(partial(
            _nucleotide_key_text_callback,
            text_object=text_object,
            nucleotide_body_patch=nucleotide_body_patch))
        subplot.add_patch(key_patch)
    return key_patches


def _draw_key_text(text: str, center, theta1, theta2, radius, width,
                   subplot: plt_axes.Subplot, **text_kwargs):
    text_center, theta = _get_wedge_center_and_angle(
        center, radius, theta1, theta2, width)
    text_angle = theta - 90
    if text_angle > 90:
        text_angle = text_angle - 180
    if len(text) > 10:
        text = text.replace("_", "_\n")
    text_object = subplot.annotate(text, xy=text_center,
                                   verticalalignment="center",
                                   horizontalalignment="center",
                                   rotation=text_angle, **text_kwargs)
    text_object.draw(subplot.figure.canvas.renderer)
    return text_object


def _get_wedge_center_and_angle(center, radius, theta1, theta2, width):
    text_radius = radius + width // 2
    theta = theta1 + (theta2 - theta1) / 2
    theta_grad = np.pi * theta / 180
    delta_x, delta_y = (np.cos(theta_grad) * text_radius, np.sin(theta_grad)
                        * text_radius)
    wedge_center = [center[0] + delta_x, center[1] + delta_y]
    return wedge_center, theta


def _nucleotide_name_callback(patch: patches.Circle,
                              text_object, x_border=0.1, y_border=0.1):
    patch_window_extent = patch.get_window_extent()
    _scale_font_size(text_object, patch_window_extent.width,
                     patch_window_extent.height, x_border, y_border)


def _nucleotide_key_text_callback(patch: patches.Wedge,
                                  nucleotide_body_patch: patches.Circle,
                                  text_object, x_border=0.2, y_border=0.1):
    # assumes that the coordinates are equal scaled in the view
    radius = nucleotide_body_patch.radius
    radius_with_width = radius + patch.width
    radius_with_half_width = radius + patch.width / 2

    initial_text_rotation = text_object.get_rotation()
    text_object.set_rotation(0)
    body_patch_window_extent = nucleotide_body_patch.get_window_extent()
    scale_pixel_to_units = body_patch_window_extent.width / 2 / radius
    wedge_height_pixel = scale_pixel_to_units * patch.width
    wedge_width = (2 * np.pi * radius_with_half_width *
                   np.abs(patch.theta2 - patch.theta1) / 360)

    text_width_max = min(
        wedge_width,
        2 * (radius_with_width ** 2 - radius_with_half_width ** 2) ** 0.5)
    text_width_max_pixel = scale_pixel_to_units * text_width_max

    _scale_font_size(text_object, text_width_max_pixel,
                     wedge_height_pixel, x_border, y_border)
    text_object.set_rotation(initial_text_rotation)


def _scale_font_size(text_object, max_width, max_height, x_border, y_border):
    text_fontsize = text_object.get_fontsize()
    text_window_extent = text_object.get_window_extent()
    fontsize_scale_width = (max_width * (1 - x_border)
                            / text_window_extent.width)
    fontsize_scale_height = (max_height * (1 - y_border)
                             / text_window_extent.height)
    fontsize_scale = min(10, fontsize_scale_width, fontsize_scale_height)
    fontsize_rescaled = max(1, int(fontsize_scale * text_fontsize))
    text_object.set_fontsize(fontsize_rescaled)


def _draw_click_instructions(subplot: plt_axes.Subplot,
                             doubleclick=True, singleclck=True):
    instruction_texts = list()
    instruction_texts.append("Interactive instructions:")
    if singleclck:
        instruction_texts.append(
            "Click once on nucleotide to see its information")
    if doubleclick:
        instruction_texts.append(
            "Make double clock on nucleotide to cut the subgraph with its "
            "incoming and outgoing nucleotides in new figure")

    instruction_text = "\n".join(instruction_texts)
    subplot.annotate(
        instruction_text, (0.5, 0.01), xycoords="figure fraction",
        ha="center", va="bottom", ma="left",
        bbox=dict(facecolor='white', edgecolor='blue', pad=5.0))


def _get_nucleotide_color(nucleotide: Nucleotide) -> Tuple[str, Optional[type]]:
    # pylint: disable=too-many-return-statements,too-many-branches
    # no way to combine it further and no way to express it over dict
    if isinstance(nucleotide, ModelPlugin):
        return "royalblue", ModelPlugin
    if isinstance(nucleotide, ModelPostProcessor):
        return "violet", ModelPostProcessor
    if isinstance(nucleotide, ModelLoss):
        return "darkorange", ModelLoss
    if isinstance(nucleotide, ModelMetric):
        return "olive", ModelMetric
    if isinstance(nucleotide, ModelSummary):
        return 'sienna', ModelSummary
    if isinstance(nucleotide, KPIEvaluatorCallback):
        return 'salmon', KPIEvaluatorCallback
    if isinstance(nucleotide, CoordinatorCallback):
        return 'cyan', CoordinatorCallback
    if isinstance(nucleotide, Dataset):
        return 'green', Dataset
    if isinstance(nucleotide, DataFeeder):
        return 'green', DataFeeder
    if isinstance(nucleotide, DataReader):
        return 'lime', DataReader
    if isinstance(nucleotide, DataProcessor):
        return 'darkgreen', DataProcessor
    if isinstance(nucleotide, PredictorNucleotide):
        return 'violet', PredictorNucleotide
    if isinstance(nucleotide, KPIAccumulator):
        return 'salmon', KPIAccumulator
    if isinstance(nucleotide, KPIPlugin):
        return 'royalblue', KPIPlugin
    return 'green', Nucleotide


def _get_key_color(key_type: str):
    if key_type == "incoming":
        return "lightgray"
    return "darkgrey"
