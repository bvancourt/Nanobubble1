"""
This file contains functions for statistical tests and visualization of the kind
of data where you basically have an un-ordered list of numbers associate with
each of two or more groups (e.g. untreated vs. treatment A vs. treatment B).
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy.stats
import pingouin
import scikit_posthocs
import pandas as pd
import statsmodels.stats.multitest


def _swarm_offsets(y, r): # helper function for swarm_plot function below.
    x = np.zeros(len(y))

    dist_contraints = np.sqrt(
        np.maximum(0, 4 * r**2 - (y[np.newaxis, :] - y[:, np.newaxis]) ** 2)
        * (1 - np.eye(len(y)))
    )
    if np.any(dist_contraints) > 0:
        unused_indices = list(np.arange(len(y)))
        used = np.zeros(len(x), dtype=bool)
        this_index = unused_indices[-1]
        while len(unused_indices) > 0:

            unused_indices.remove(this_index)
            if not np.any(dist_contraints[this_index, :] > 0):
                x[this_index] = 0
                used[this_index] = True

                if len(unused_indices) > 0:
                    this_index = unused_indices[-1]

            elif not np.any((dist_contraints[this_index, :] > 0) & used):
                used[this_index] = True

                next_index = np.argmax(
                    np.float32((dist_contraints[this_index, :] > 0) & ~used)
                    / (1 + dist_contraints[this_index, :])
                )

                dmin = dist_contraints[this_index, next_index]

                sign_flip = np.random.choice([-1, 1])
                x[this_index] = -sign_flip * dmin / 2
                if not np.any(
                    np.abs(sign_flip * dmin / 2 - x[used])
                    < dist_contraints[next_index, used]
                ):

                    x[next_index] = sign_flip * dmin / 2

                    used[next_index] = True
                    unused_indices.remove(next_index)
                else:
                    x[this_index] = 0

                if len(unused_indices) > 0:
                    this_index = np.argmax(
                        np.float32((dist_contraints[this_index, :] > 0) & ~used)
                        / (1 + dist_contraints[this_index, :])
                        - used
                    )
            else:
                right_loc = np.max(dist_contraints[this_index, used] + x[used])
                left_loc = np.min(-dist_contraints[this_index, used] + x[used])
                candidate_locations = [left_loc, right_loc]

                x[this_index] = candidate_locations[
                    np.argmin(np.abs(candidate_locations))
                ]
                used[this_index] = True

                if len(unused_indices) > 0:
                    this_index = np.argmax(
                        np.float32(~used) / (1 + dist_contraints[this_index, :])
                    )

            x[np.sum(dist_contraints, axis=0) == 0] = np.mean(
                x[np.sum(dist_contraints, axis=0) != 0]
            )
            x[used] -= np.median(x[used])
        x -= np.mean(x)
    return x


def _sort_comp_bars(bars, data_tops): # helper for swarm_plot function below.
    lower_bounds = []
    for bar in bars:
        affected_cols = np.arange(
            np.minimum(bar[0], bar[1]), np.maximum(bar[0], bar[1]) + 1
        )
        lower_bounds.append(np.max(data_tops[affected_cols]))

    order = np.argsort(lower_bounds)

    return [bars[index] for index in order]


def swarm_plot(
    data,
    exclusion_masks=None,
    point_colors=None,
    excluded_data=None,
    always_show_excluded=True,
    comp_bars=None,
    dot_radius=0.03,
    margin=0.05,
    h_margin_factor=2,
    ax_height=2.5,
    col_labels=None,
    h_lines=None,
    force_y_min=None,
    colors=None,
    excluded_color=(0.7, 0.7, 0.7),
    wide_bars="median",
    narrow_bars="quartiles",
    include_whiskers=True,
    box_width="auto",
    wide_bar_linewidth=1,
    narrow_bar_linewidth=1,
    dpi=600,
    title=None,
    axis_label=None,
    min_col_spacing=1.5,
    force_min_col_spacing=False,
    bar_text_bump_mode="safe",
    title_color=(0, 0, 0),
    fig_save_path=None,
    color_col_labels=False,
    no_bottom_margin=False,
    bar_from_zero=False,
    filled_bar_alpha=0,
    pair_lines=False,
    label_rot=0,
    label_ha="center",
    open_axis=False,
):
    if len(data) == 0:
        print("Cannot make plot from no data!")
        return

    if colors == None:
        cmap = matplotlib.cm.get_cmap("tab10")
        colors = [np.array(cmap(i % 10))[:3] for i in range(len(data))]

    if (excluded_data == None) and not (exclusion_masks == None):
        excluded_data = [
            col[excl_mask] for col, excl_mask in zip(data, exclusion_masks)
        ]
        data = [
            col[~excl_mask] for col, excl_mask in zip(data, exclusion_masks)
        ]

    elif not ((excluded_data == None) or (exclusion_masks == None)):
        print(
            "Ignoring provided exclusion masks, because excluded data list was "
            + "provided. Using both is not supported."
        )

    if point_colors == None:
        point_colors = [
            np.ones((len(col), 1, 3)) * colors[i]
            for (i, col) in enumerate(data)
        ]

    elif exclusion_masks is not None:  
        # if exclusion masks are provided, assume that they apply to both data 
        # and point colors
        point_colors = [
            pc_col[~excl_mask, :]
            for pc_col, excl_mask in zip(point_colors, exclusion_masks)
        ]

    if h_lines == None:
        h_lines = []

    tops = [np.max(col) for col in data]
    bottoms = [np.min(col) for col in data]

    if always_show_excluded and not (excluded_data == None):
        for excl, old_top, old_bottom in zip(excluded_data, tops, bottoms):
            if len(excl) > 0:
                old_top = np.maximum(np.max(excl), old_top)
                old_bottom = np.minimum(np.min(excl), old_bottom)

    if not always_show_excluded:
        excluded_data = [
            excl[
                (excl < (top + 0.2 * (top - bottom)))
                & (excl > (bottom - 0.2 * (top - bottom)))
            ]
            for excl, top, bottom in zip(excluded_data, tops, bottoms)
        ]

    if not (excluded_data == None):
        for i, (excl, top, bottom) in enumerate(
            zip(excluded_data, tops, bottoms)
        ):
            if np.prod(excl.shape):
                tops[i] = np.maximum(top, np.max(excl))
                bottoms[i] = np.minimum(bottom, np.min(excl))

    effective_data_tops = np.array(tops)
    i_highest_col = np.argmax(tops)

    data_min = np.min(bottoms)
    if not (force_y_min == None):
        print(f"forced effective data minimum to {force_y_min}.")
        data_min = float(force_y_min)

    if not comp_bars == None:
        for shrink_it in range(10):
            candidate_y_scales = []
            for i in range(len(data)):
                candidate_y_scales.append(
                    (effective_data_tops[i] - data_min)
                    / (
                        1
                        - margin
                        * (
                            1
                            - margin * (1 if no_bottom_margin else 2)
                            + 1.5 * len(comp_bars)
                        )
                    )
                )

            if np.any(np.isinf(np.array(candidate_y_scales))):
                print("calculated invalid vertical scale -> shrinking margin.")
                margin /= 2
                h_margin_factor *= 2
                if shrink_it == 9:
                    warnings.warn(
                        "exceded margin shrink iterations"
                        + "swarm_plot() may not be working properly."
                    )
            else:
                break

        global_y_scale = np.max(candidate_y_scales)
        effective_data_tops += dot_radius * global_y_scale

        sorted_bars = _sort_comp_bars(comp_bars, np.array(tops))
        bar_heights = np.zeros(len(sorted_bars))
        left_leg_heights = np.zeros(len(sorted_bars))
        right_leg_heights = np.zeros(len(sorted_bars))

        for i, bar in enumerate(sorted_bars):
            affected_cols = np.arange(
                np.minimum(bar[0], bar[1]), np.maximum(bar[0], bar[1]) + 1
            )
            effective_data_tops[affected_cols] += 0.5 * margin * global_y_scale

            bar_heights[i] = (
                np.max(effective_data_tops[affected_cols])
                + 0.5 * margin * global_y_scale
            )
            left_leg_heights[i] = effective_data_tops[
                np.minimum(bar[0], bar[1])
            ]
            right_leg_heights[i] = effective_data_tops[
                np.maximum(bar[0], bar[1])
            ]

            effective_data_tops[affected_cols] = (
                np.max(effective_data_tops[affected_cols])
                + margin * global_y_scale * 0.5
            )

            if (bar_text_bump_mode in ["exact", "safe"]) and not (bar[2] == ""):
                if len(affected_cols) % 2 == 1:
                    text_bump_col = int(np.mean(affected_cols))
                    effective_data_tops[text_bump_col] += (
                        margin * global_y_scale * 0.75
                    )
                elif len(affected_cols) % 2 == 0 and (
                    bar_text_bump_mode == "safe"
                ):
                    text_bump_col = int(np.floor(np.mean(affected_cols)))
                    effective_data_tops[text_bump_col] += (
                        margin * global_y_scale * 0.75
                    )
                    effective_data_tops[text_bump_col + 1] += (
                        margin * global_y_scale * 0.75
                    )
            elif bar_text_bump_mode == "all" and not (bar[2] == ""):
                effective_data_tops[affected_cols] += (
                    margin * global_y_scale * 0.75
                )

    else:
        for shrink_it in range(10):
            candidate_y_scales = []
            for i in range(len(data)):
                candidate_y_scales.append(
                    (effective_data_tops[i] - data_min)
                    / (1 - margin * (1 if no_bottom_margin else 2))
                )

            if np.any(np.isinf(np.array(candidate_y_scales))):
                print("calculated invalid vertical scale -> shrinking margin.")
                margin /= 2
                h_margin_factor *= 2
                if shrink_it == 9:
                    warnings.warn(
                        "exceded margin shrink iterations. swarm plot may not "
                        + "be working."
                    )
            else:
                break

        global_y_scale = np.nanmax(candidate_y_scales)
        effective_data_tops += dot_radius * global_y_scale

    i_highest_col = np.argmax(effective_data_tops)

    global_y_min = (
        data_min
        if no_bottom_margin
        else data_min - (margin + dot_radius) * global_y_scale
    )
    global_y_max = (
        effective_data_tops + (margin + dot_radius) * global_y_scale
    )[i_highest_col]

    col_x_vals = [
        _swarm_offsets((col - np.mean(col)) / global_y_scale, dot_radius)
        for col in data
    ]

    global_x_max = np.sum(
        [
            np.maximum(
                (np.max(cx) - np.min(cx)) * (~force_min_col_spacing),
                min_col_spacing * dot_radius,
            )
            for cx in col_x_vals
        ]
    ) + (len(data)) * (2 * margin * h_margin_factor + 2 * dot_radius)

    col_shifts = np.zeros(len(data))
    col_shifts[0] = (
        margin * h_margin_factor
        + dot_radius
        - np.minimum(np.min(col_x_vals[0]), -dot_radius * min_col_spacing / 2)
    )

    # horizontally center the col
    first_col_width = np.maximum(
        (np.max(col_x_vals[0]) - np.min(col_x_vals[0]))
        * (~force_min_col_spacing),
        min_col_spacing * dot_radius,
    )
    last_col_width = np.maximum(
        (np.max(col_x_vals[-1]) - np.min(col_x_vals[-1]))
        * (~force_min_col_spacing),
        min_col_spacing * dot_radius,
    )
    asymmetry = np.abs((last_col_width - first_col_width) / 2)

    global_x_max += asymmetry
    if first_col_width < last_col_width:  # necessary?
        col_shifts += asymmetry

    if len(data) > 1:
        for i in range(1, len(data)):
            col_shifts[i] = (
                col_shifts[i - 1]
                + np.maximum(
                    min_col_spacing * dot_radius,
                    (np.max(col_x_vals[i - 1]) - np.min(col_x_vals[i]))
                    * (~force_min_col_spacing),
                )
                + margin * 2 * h_margin_factor
                + 2 * dot_radius
            )

    if (global_x_max - col_shifts[-1]) < col_shifts[0]:
        global_x_max = col_shifts[-1] + col_shifts[0]

    fig = plt.figure(
        dpi=dpi, figsize=[1.5 * ax_height * global_x_max, 1.5 * ax_height]
    )
    ax = fig.add_axes((1 / 6, 1 / 6, 5 / 6, 5 / 6))

    # plot background grid lines
    if h_lines == "means":
        for y, c in zip([np.mean(col) for col in data], colors):
            plt.plot(
                [0, global_x_max],
                [y, y],
                color=np.array(list(np.squeeze(c)) + [0.5]),
                zorder=-1,
                linewidth=0.75,
            )
    if h_lines == "medians":
        for y, c in zip([np.median(col) for col in data], colors):
            plt.plot(
                [0, global_x_max],
                [y, y],
                color=np.array(list(np.squeeze(c)) + [0.5]),
                zorder=-1,
                linewidth=0.75,
            )
    elif type(h_lines) is list:
        for y in h_lines:
            plt.plot(
                [0, global_x_max],
                [y, y],
                color=(0.8, 0.8, 0.8),
                zorder=-1,
                linewidth=0.75,
            )

    # plot the swarm of dots
    for x, s, y, c in zip(col_x_vals, col_shifts, data, point_colors):
        for xp, yp, cp in zip(x, y, c):
            plt.scatter(
                xp + s,
                yp,
                s=(140 * dot_radius * ax_height) ** 2,
                color=cp.reshape((1, 1, 3)),
            )

    if not excluded_data == None:
        for s, y in zip(col_shifts, excluded_data):
            if len(y) > 0:
                x = _swarm_offsets(
                    (y - np.mean(y)) / global_y_scale, dot_radius / 2
                )
                for xp, yp in zip(x, y):
                    plt.scatter(
                        xp + s,
                        yp,
                        s=(70 * dot_radius * ax_height) ** 2,
                        color=excluded_color,
                    )

    if not wide_bars == None:
        if box_width == "auto":
            bar_widths = [
                np.std(x) * 1.5 + dot_radius * ax_height for x in col_x_vals
            ]
        else:
            bar_widths = [box_width * dot_radius * ax_height] * len(col_x_vals)
        if wide_bars == "mean":
            wide_bar_heights = [np.mean(y) for y in data]
        elif wide_bars == "median":
            wide_bar_heights = [np.median(y) for y in data]
        else:
            raise ValueError(
                f"wide_bars should be 'mean' or 'median' not {wide_bars}."
            )
        for i_col, shift in enumerate(col_shifts):
            plt.plot(
                [shift - bar_widths[i_col], shift + bar_widths[i_col]],
                [wide_bar_heights[i_col]] * 2,
                color="black",
                solid_capstyle="round",
                linewidth=wide_bar_linewidth,
            )
            if bar_from_zero:
                plt.plot(
                    [shift - bar_widths[i_col]] * 2,
                    [0, wide_bar_heights[i_col]],
                    color="black",
                    solid_capstyle="round",
                    linewidth=wide_bar_linewidth,
                )
                plt.plot(
                    [shift + bar_widths[i_col]] * 2,
                    [0, wide_bar_heights[i_col]],
                    color="black",
                    solid_capstyle="round",
                    linewidth=wide_bar_linewidth,
                )
            if filled_bar_alpha > 0:
                plt.fill(
                    [
                        shift - bar_widths[i_col],
                        shift + bar_widths[i_col],
                        shift + bar_widths[i_col],
                        shift - bar_widths[i_col],
                    ],
                    [wide_bar_heights[i_col], wide_bar_heights[i_col], 0, 0],
                    color=colors[i_col],
                    alpha=filled_bar_alpha,
                )

    if not narrow_bars == None:
        if box_width == "auto":
            bar_widths = [
                np.std(x) * 0.75 + dot_radius * ax_height / 2
                for x in col_x_vals
            ] * 2
        else:
            bar_widths = [box_width * dot_radius * ax_height / 2] * (
                2 * len(col_x_vals)
            )
        if narrow_bars in ["SEM", "sem", "STD", "std"]:
            if narrow_bars in ["SEM", "sem"]:
                narrow_bar_heights = [
                    np.mean(y) - np.std(y) / len(y) for y in data
                ] + [np.mean(y) + np.std(y) / len(y) for y in data]
            elif narrow_bars in ["STD", "std"]:
                narrow_bar_heights = [np.mean(y) - np.std(y) for y in data] + [
                    np.mean(y) + np.std(y) for y in data
                ]
            for i_col, shift in enumerate(col_shifts):
                # Draw a  vertical line connecting the narrow bars
                plt.plot(
                    [shift] * 2,
                    [
                        narrow_bar_heights[i_col],
                        narrow_bar_heights[i_col + len(data)],
                    ],
                    color="black",
                    solid_capstyle="round",
                    linewidth=narrow_bar_linewidth,
                )
        elif narrow_bars == "quartiles":
            narrow_bar_heights = [np.quantile(y, 0.25) for y in data] + [
                np.quantile(y, 0.75) for y in data
            ]
            for i_col, shift in enumerate(col_shifts):
                # Draw a PAIR of vertical lines connecting the narrow bars
                plt.plot(
                    [shift - bar_widths[i_col]] * 2,
                    [
                        narrow_bar_heights[i_col],
                        narrow_bar_heights[i_col + len(data)],
                    ],
                    color="black",
                    solid_capstyle="round",
                    linewidth=narrow_bar_linewidth,
                )
                plt.plot(
                    [shift + bar_widths[i_col]] * 2,
                    [
                        narrow_bar_heights[i_col],
                        narrow_bar_heights[i_col + len(data)],
                    ],
                    color="black",
                    solid_capstyle="round",
                    linewidth=narrow_bar_linewidth,
                )
                if include_whiskers:
                    plt.plot(
                        [shift] * 2,
                        [
                            np.max(data[i_col]),
                            narrow_bar_heights[i_col + len(data)],
                        ],
                        color="black",
                        solid_capstyle="round",
                        linewidth=narrow_bar_linewidth,
                    )
                    plt.plot(
                        [shift] * 2,
                        [narrow_bar_heights[i_col], np.min(data[i_col])],
                        color="black",
                        solid_capstyle="round",
                        linewidth=narrow_bar_linewidth,
                    )

        for i_col, shift in enumerate(col_shifts):
            plt.plot(
                [shift - bar_widths[i_col], shift + bar_widths[i_col]],
                [narrow_bar_heights[i_col]] * 2,
                color="black",
                solid_capstyle="round",
                linewidth=narrow_bar_linewidth,
            )
            if len(narrow_bar_heights) == 2 * len(data):
                plt.plot(
                    [shift - bar_widths[i_col], shift + bar_widths[i_col]],
                    [narrow_bar_heights[i_col + len(data)]] * 2,
                    color="black",
                    solid_capstyle="round",
                    linewidth=narrow_bar_linewidth,
                )

    if not comp_bars == None:
        for i, bar in enumerate(sorted_bars):
            ax.plot(
                [
                    col_shifts[np.minimum(bar[0], bar[1])],
                    col_shifts[np.maximum(bar[0], bar[1])],
                ],
                [bar_heights[i]] * 2,
                color="black",
                solid_capstyle="round",
            )
            ax.plot(
                [
                    col_shifts[np.minimum(bar[0], bar[1])],
                    col_shifts[np.minimum(bar[0], bar[1])],
                ],
                [bar_heights[i], left_leg_heights[i]],
                color="black",
                solid_capstyle="round",
            )
            ax.plot(
                [
                    col_shifts[np.maximum(bar[0], bar[1])],
                    col_shifts[np.maximum(bar[0], bar[1])],
                ],
                [bar_heights[i], right_leg_heights[i]],
                color="black",
                solid_capstyle="round",
            )
            ax.text(
                np.mean(
                    [
                        col_shifts[np.minimum(bar[0], bar[1])],
                        col_shifts[np.maximum(bar[0], bar[1])],
                    ]
                ),
                bar_heights[i]
                + margin * global_y_scale * (0.7 - 0.25 * (bar[2][-1] == "*")),
                bar[2],
                horizontalalignment="center",
                verticalalignment="center_baseline",
                fontweight=("bold" if bar[2][-1] == "*" else "normal"),
                fontsize=("medium" if bar[2][-1] == "*" else "xx-small"),
            )

    if open_axis == True:
        ax.spines[["right", "top"]].set_visible(False)

    if pair_lines == True:
        for i, y_vals in enumerate(zip(*data)):
            x_vals = [
                col_shifts[j] + col_x_vals[j][i] for j in range(len(data))
            ]

            ax.plot(
                x_vals,
                y_vals,
                color="black",
                solid_capstyle="round",
                linewidth=narrow_bar_linewidth / 2,
                zorder=2,
            )

    ax.set_ylim([global_y_min, global_y_max])
    ax.set_xlim([0, global_x_max])

    if not title == None:
        ax.set_title(title, color=title_color)

    if not axis_label == None:
        ax.set_ylabel(axis_label)

    ax.set_yticks(
        ax.get_yticks()[
            (ax.get_yticks() < np.max(tops) + 0.2 * global_y_scale)
            & (ax.get_yticks() >= global_y_min)
        ]
    )

    if not col_labels == None:
        ax.set_xticks(col_shifts)
        ax.set_xticklabels(col_labels, rotation=label_rot, ha=label_ha)
        if color_col_labels == True:
            for i, tick_text in enumerate(ax.get_xticklabels()):
                tick_text.set_color(colors[i])

    else:
        ax.set_xticks([])

    if not fig_save_path == None:
        plt.savefig(fig_save_path, bbox_inches="tight", transparent=True)

    plt.show()


class StatResult:
    def __init__(self, p=np.nan, statistic=np.nan):
        self.p = p
        self.pvalue = p
        self.statistic = statistic


def make_comparisons(
    data,
    parametric=True,
    equal_variances=True,
    post_hoc=None,
    outlier_detection="Grubbs",
    outlier_test_min_N=4,
    strict_outlier_test=False,
    alpha=0.05,
    pairs_to_check=None,
    group_names=None,
    multiple_comparisons="Tukey",
    include_near_sig=True,
    common_control=None,
):

    n_groups = len(data)
    if n_groups == 0:
        return [], [], {}, [], []

    if pairs_to_check == None:  # Default is to compare all pairs, if applicable
        pairs_to_check = []
        for i in range(n_groups):
            for j in range(i):
                pairs_to_check.append((i, j))

    analysis_log = {}

    if group_names == None:
        group_names = [f"Column {i}" for i in range(n_groups)]

    if outlier_detection in [None, "none"]:
        included_data = data
        excluded_data = [np.empty(0)] * n_groups
        analysis_log["Outlier Test"] = None

    elif outlier_detection == "IQR":
        medians = [np.median(col) for col in data]
        IQRs = [np.quantile(col, 0.75) - np.quantile(col, 0.25) for col in data]
        included_data = []
        excluded_data = []
        analysis_log["Outlier Test"] = {
            "test_type": "IQR",
            "IQR threshold": 3 if strict_outlier_test else 1.5,
            "exclusion_masks": [],
        }

        for i_col, (col, median, IQR) in enumerate(zip(data, medians, IQRs)):
            if len(col) >= outlier_test_min_N:
                exclusion_mask = (
                    np.abs(col - median)
                    > (2.5 if strict_outlier_test else 1.5) * IQR
                )
                included_data.append(col[~exclusion_mask])
                excluded_data.append(col[exclusion_mask])
                analysis_log["Outlier Test"]["exclusion_masks"].append(
                    exclusion_mask
                )
            else:
                analysis_log["Outlier Test"]["exclusion_masks"].append(
                    np.zeros(len(col))
                )
                included_data.append(col)
                excluded_data.append(np.empty(0))

        analysis_log["Outlier Test"]["n_points_excluded"] = np.sum(
            [
                np.sum(mask)
                for mask in analysis_log["Outlier Test"]["exclusion_masks"]
            ]
        )

    elif outlier_detection == "Grubbs":
        alpha_grubbs = 0.01 if strict_outlier_test else 0.05

        analysis_log["Outlier Test"] = {
            "test_type": "Grubbs",
            "alpha": alpha_grubbs,
            "exclusion_masks": [],
        }
        included_data = []
        excluded_data = []
        outlier_masks = [np.zeros(len(col), dtype=bool) for col in data]

        for col, outliers in zip(data, outlier_masks):
            N = len(col[~outliers])
            found_no_outliers = False
            while N >= outlier_test_min_N:
                z_scores = (col[~outliers] - np.mean(col[~outliers])) / np.std(
                    col[~outliers], ddof=1
                )
                i_extreme = np.arange(len(col))[~outliers][
                    np.argmax(np.abs(z_scores))
                ]
                t_squared = (
                    scipy.stats.t.ppf(1 - alpha_grubbs / (2 * N), N - 2) ** 2
                )
                critical_point = ((N - 1) / np.sqrt(N)) * np.sqrt(
                    t_squared / (N - 2 + t_squared)
                )

                if np.max(np.abs(z_scores)) > critical_point:
                    outliers[i_extreme] = True
                    N = len(col[~outliers])
                else:
                    found_no_outliers = True
                    break

            included_data.append(col[~outliers])
            excluded_data.append(col[outliers])

        analysis_log["Outlier Test"]["exclusion_masks"] = outlier_masks
        analysis_log["Outlier Test"]["n_points_excluded"] = np.sum(
            [
                np.sum(mask)
                for mask in analysis_log["Outlier Test"]["exclusion_masks"]
            ]
        )

    if (
        n_groups == 1
    ):  # For only one group, there is nothing to copare, 
        # so we can just return the outlier test results now.
        return included_data, excluded_data, analysis_log, [], []

    default_res = StatResult()
    normality_results = [
        (scipy.stats.shapiro(col) if len(col) > 2 else default_res)
        for col in included_data
    ]
    analysis_log["Normality Test"] = {
        "test_type": "Shapiro-Wilk",
        "normality_statistcis": [res.statistic for res in normality_results],
        "p-values": [res.pvalue for res in normality_results],
        "significant?": np.any(
            [res.pvalue < 0.05 / n_groups for res in normality_results]
        ),
    }

    follow_up_tests_warranted = False
    significant_pairs = []
    significant_p_values = []

    if parametric == True:
        # ANOVA or T-test

        # Starts by testing for unequal variances
        chi_sq, p = scipy.stats.bartlett(*included_data)
        analysis_log["Equal Variances Test"] = {
            "test_type": "Bartlett's",
            "statistic": chi_sq,
            "p": p,
            "significant?": p < 0.05,
        }

        if (
            equal_variances == "auto"
        ):  # Actually, this does ANOVA with AND without Welch's correction, 
            # just in case.
            if p < 0.05:
                equal_variances = False

                if n_groups > 2:
                    F, p = scipy.stats.f_oneway(*included_data)
                    analysis_log["Alternative Group Differences Test"] = {
                        "test_type": "ANOVA",
                        "F": F,
                        "p": p,
                        "significant?": p < alpha,
                    }
                elif (
                    n_groups == 2
                ):  # For only two groups, give the t statistic rather than the 
                    #   F statistic.
                    t, p = scipy.stats.ttest_ind(*included_data)
                    analysis_log["Alternative Group Differences Test"] = {
                        "test_type": "T-test",
                        "t": t,
                        "p": p,
                        "significant?": p < alpha,
                    }
                    if p < (alpha if not include_near_sig else 2 * alpha):
                        significant_pairs.append((0, 1))
                        significant_p_values.append(p)

            else:
                equal_variances = True
                if n_groups > 2:  # Welch's ANOVA (just in case)
                    df_dict = {"i_group": [], "data": []}
                    for i, col in enumerate(included_data):
                        df_dict["i_group"] += [np.float32(i)] * len(col)
                        df_dict["data"] += list(col)

                    pg_res = pingouin.welch_anova(
                        dv="data", between="i_group", data=pd.DataFrame(df_dict)
                    )
                    analysis_log["Group Differences Test"] = {
                        "test_type": "Welch's ANOVA",
                        "F": pg_res["F"][0],
                        "p": pg_res["p-unc"][0],
                        "ddof1": pg_res["ddof1"][0],
                        "ddof2": pg_res["ddof2"][0],
                        "np2": pg_res["np2"][0],
                        "significant?": pg_res["p-unc"][0] < alpha,
                    }

                elif n_groups == 2:
                    t, p = scipy.stats.ttest_ind(
                        *included_data, equal_var=False
                    )
                    analysis_log["Group Differences Test"] = {
                        "test_type": "Welch's T-test",
                        "t": t,
                        "p": p,
                        "significant?": p < alpha,
                    }
                    if p < (alpha if not include_near_sig else 2 * alpha):
                        significant_pairs.append((0, 1))
                        significant_p_values.append(p)

        if equal_variances == True:
            # Standard ANOVA/t-test
            if n_groups > 2:
                F, p = scipy.stats.f_oneway(*included_data)
                analysis_log["Group Differences Test"] = {
                    "test_type": "ANOVA",
                    "F": F,
                    "p": p,
                    "significant?": p < alpha,
                }
            elif (
                n_groups == 2
            ):  # For only two groups, give the t statistic rather than the F 
                #  statistic.
                t, p = scipy.stats.ttest_ind(*included_data)
                analysis_log["Group Differences Test"] = {
                    "test_type": "T-test",
                    "t": t,
                    "p": p,
                    "significant?": p < alpha,
                }
                if p < (alpha if not include_near_sig else 2 * alpha):
                    significant_pairs.append((0, 1))
                    significant_p_values.append(p)

        elif equal_variances == False:
            # have to make data frame for Pingouin, since SciPy did not have 
            #   this test.
            df_dict = {"i_group": [], "data": []}
            for i, col in enumerate(included_data):
                df_dict["i_group"] += [np.float32(i)] * len(col)
                df_dict["data"] += list(col)

            pg_res = pingouin.welch_anova(
                dv="data", between="i_group", data=pd.DataFrame(df_dict)
            )
            analysis_log["Group Differences Test"] = {
                "test_type": "Welch's ANOVA",
                "F": pg_res["F"][0],
                "p": pg_res["p-unc"][0],
                "ddof1": pg_res["ddof1"][0],
                "ddof2": pg_res["ddof2"][0],
                "np2": pg_res["np2"][0],
                "significant?": pg_res["p-unc"][0] < alpha,
            }

        else:
            raise ValueError(
                'Did not compare groups! "equal_variances" must be True, False,'
                + ' or "auto" for Parametric mode.'
            )

    elif parametric == False:
        # Kuskal-Wallis or Mann-Whitney rank comparison tests
        if n_groups > 2:
            # Kuskal-Wallis
            if not np.var(np.hstack(included_data)) == 0:
                H, p = scipy.stats.kruskal(*included_data)
            else:
                H, p = np.nan, np.nan
            analysis_log["Group Differences Test"] = {
                "test_type": "Kustal-Wallis",
                "H": H,
                "p": p,
                "significant?": p < alpha,
            }

        elif n_groups == 2:
            # Mann-Whitney
            U, p = scipy.stats.mannwhitneyu(*included_data)
            analysis_log["Group Differences Test"] = {
                "test_type": "Mann-Whitney",
                "U": U,
                "p": p,
                "significant?": p < alpha,
            }
            if p < (alpha if not include_near_sig else 2 * alpha):
                significant_pairs.append((0, 1))
                significant_p_values.append(p)
        else:
            print('"parametric" must be True or False...')

    if (
        (analysis_log["Group Differences Test"]["significant?"] == True)
        and not (multiple_comparisons in [False, None])
        and (n_groups > 2)
    ):

        if multiple_comparisons == "Dunn":
            useable_group_indices = np.squeeze(
                np.where([len(col) >= 2 for col in included_data])
            )

            if len(useable_group_indices) > 2:

                useable_data = [included_data[i] for i in useable_group_indices]
                if not np.var([np.mean(col) for col in useable_data]) == 0:
                    p_dunn = np.array(
                        scikit_posthocs.posthoc_dunn(
                            useable_data, sort=False
                        )
                    )
                    p_matrix = np.full((n_groups, n_groups), np.nan)
                    for i_i, i in enumerate(useable_group_indices):
                        for i_j, j in enumerate(useable_group_indices[:i_i]):
                            if not i == j:
                                p_matrix[i, j] = p_dunn[i_i, i_j]
                                p_matrix[j, i] = p_matrix[i, j]
                                if p_matrix[i, j] < alpha * (
                                    include_near_sig + 1
                                ):
                                    if ((i, j) in pairs_to_check) or (
                                        (j, i) in pairs_to_check
                                    ):
                                        significant_pairs.append((i, j))
                                        significant_p_values.append(
                                            p_matrix[i, j]
                                        )

                    analysis_log["Individual Comparisons"] = {
                        "test_type": "Dunn",
                        "p": p_matrix,
                        "significant?": np.any(p_matrix < alpha),
                    }

        if multiple_comparisons == "Tukey":
            useable_group_indices = np.squeeze(
                np.where([len(col) >= 2 for col in included_data])
            )
            assume_equal_variances = (
                analysis_log["Equal Variances Test"]["significant?"] == True
            ) or equal_variances
            if len(useable_group_indices) > 2 and assume_equal_variances:
                useable_data = [included_data[i] for i in useable_group_indices]

                tukey_res = scipy.stats.tukey_hsd(*useable_data)

                p_matrix = np.full((n_groups, n_groups), np.nan)
                for i_i, i in enumerate(useable_group_indices):
                    for i_j, j in enumerate(useable_group_indices[:i_i]):
                        if not i == j:
                            p_matrix[i, j] = tukey_res.pvalue[i_i, i_j]
                            p_matrix[j, i] = p_matrix[i, j]
                            if p_matrix[i, j] < alpha * (include_near_sig + 1):
                                significant_pairs.append((i, j))
                                significant_p_values.append(p_matrix[i, j])

                analysis_log["Individual Comparisons"] = {
                    "test_type": "Tukey-Kramer",
                    "p": p_matrix,
                    "significant?": np.any(p_matrix < alpha),
                }

    return (
        included_data,
        excluded_data,
        analysis_log,
        significant_pairs,
        significant_p_values,
    )

    # else:
    #    return included_data, excluded_data, analysis_log, [], []


def multiple_pairwise_comparisons(
    data,
    tests="Mann-Whitney",
    alpha=0.05,
    pairs_to_check=None,
    p_adjust_method="Holm",
    outlier_detection="Grubbs",
    outlier_test_min_N=4,
    include_near_sig=True,
    strict_outlier_test=False,
):
    n_groups = len(data)
    if n_groups == 0:
        print("Cannot analyze zero columns of data.")
        return {}

    if pairs_to_check == None:  # Default is to compare all pairs, if applicable
        pairs_to_check = []
        for i in range(n_groups):
            for j in range(i):
                pairs_to_check.append((i, j))

    results_dict = {
        "tests": tests,
        "alpha": alpha,
        "p_adjust_method": p_adjust_method,
        "outlier_detection": outlier_detection,
        "pairs_compared": pairs_to_check,
    }

    if type(tests) is str:
        tests = [tests] * len(pairs_to_check)

    elif (type(tests) is list) and not (len(tests) == len(pairs_to_check)):
        raise ValueError(
            '"tests" must either be a single string or a list the same length '
            + 'as "data".'
        )
        return {}

    if outlier_detection in [None, "none"]:
        included_data = data
        excluded_data = [np.empty(0)] * n_groups

    elif outlier_detection == "IQR":
        medians = [np.median(col) for col in data]
        IQRs = [np.quantile(col, 0.75) - np.quantile(col, 0.25) for col in data]
        included_data = []
        excluded_data = []
        results_dict["exclusion_masks"] = []

        for i_col, (col, median, IQR) in enumerate(zip(data, medians, IQRs)):
            if len(col) >= outlier_test_min_N:
                results_dict["IQR_thresh"] = 2.5 if strict_outlier_test else 1.5
                exclusion_mask = (
                    np.abs(col - median)
                    > (2.5 if strict_outlier_test else 1.5) * IQR
                )
                included_data.append(col[~exclusion_mask])
                excluded_data.append(col[exclusion_mask])
                results_dict["exclusion_masks"].append(exclusion_mask)
            else:
                results_dict["exclusion_masks"].append(
                    np.zeros(len(col), dtype=bool)
                )
                included_data.append(col)
                excluded_data.append(np.empty(0))

    elif outlier_detection == "Grubbs":
        alpha_grubbs = 0.01 if strict_outlier_test else 0.05

        results_dict["alpha_grubbs"] = alpha_grubbs

        included_data = []
        excluded_data = []
        outlier_masks = [np.zeros(len(col), dtype=bool) for col in data]

        for col, outliers in zip(data, outlier_masks):
            N = len(col[~outliers])
            found_no_outliers = False
            while N >= outlier_test_min_N:
                z_scores = (col[~outliers] - np.mean(col[~outliers])) / np.std(
                    col[~outliers], ddof=1
                )
                i_extreme = np.arange(len(col))[~outliers][
                    np.argmax(np.abs(z_scores))
                ]
                t_squared = (
                    scipy.stats.t.ppf(1 - alpha_grubbs / (2 * N), N - 2) ** 2
                )
                critical_point = ((N - 1) / np.sqrt(N)) * np.sqrt(
                    t_squared / (N - 2 + t_squared)
                )

                if np.max(np.abs(z_scores)) > critical_point:
                    outliers[i_extreme] = True
                    N = len(col[~outliers])
                else:
                    found_no_outliers = True
                    break

            included_data.append(col[~outliers])
            excluded_data.append(col[outliers])

        results_dict["exclusion_masks"] = outlier_masks

    results_dict["included_data"] = included_data
    results_dict["excluded_data"] = excluded_data

    if (
        n_groups == 1
    ):  # For only one group, there is nothing to copare, so we can just return 
        # the outlier test results now.
        return results_dict

    # Do Comparisons
    p_values = []
    statistics = []
    for (i, j), test in zip(pairs_to_check, tests):
        if test in ["none", None]:
            p_values.append(np.nan)
            statistics.append(np.nan)
        elif test == "Mann-Whitney":
            U, p = scipy.stats.mannwhitneyu(included_data[i], included_data[j])
            p_values.append(p)
            statistics.append(U)
        elif test == "T-test":
            t, p = scipy.stats.ttest_ind(included_data[i], included_data[j])
            p_values.append(p)
            statistics.append(t)
        elif test == "Paired T-test":
            t, p = scipy.stats.ttest_rel(included_data[i], included_data[j])
            print(t, p)
            p_values.append(p)
            statistics.append(t)
        elif test in ["Welch", "Welch's T-test"]:
            t, p = scipy.stats.ttest_ind(
                included_data[i], included_data[j], equal_var=False
            )
            p_values.append(p)
            statistics.append(t)
        else:
            p_values.append(f'unrecognized test "{test}"')
            statistics.append(f'unrecognized test "{test}"')

    results_dict["p_values"] = p_values

    if p_adjust_method in ["none", None]:
        p_adj = p_values
        is_significant = [p < alpha for p in p_values]
    else:
        not_significant, p_adj, alpha_sidak, alpha_bonferroni = (
            statsmodels.stats.multitest.multipletests(
                p_values, method=p_adjust_method, alpha=alpha
            )
        )
        is_significant = [ns == False for ns in not_significant]
        results_dict["p_adj"] = p_adj
        results_dict["alpha_sidak"] = alpha_sidak
        results_dict["alpha_bonferroni"] = alpha_bonferroni

    results_dict["significant?"] = is_significant

    significant_pairs = []
    sig_pair_p_values = []
    for (i, j), q in zip(pairs_to_check, p_adj):
        if q < (alpha if not include_near_sig else 2 * alpha):
            significant_pairs.append((i, j))
            sig_pair_p_values.append(q)

    results_dict["sig_pairs"] = significant_pairs
    results_dict["sig_pair_p_values"] = sig_pair_p_values

    return results_dict


def p_val_to_disp_string(
    p,
    alpha=0.05,
    include_ns=False,
    p_for_near_sig=False,
    mode="asterisks",
    value_name="p",
    max_stars=4,
):
    # For values that are not statistically significant
    if p > 2 * alpha:
        return "n.s." if include_ns else ""
    elif p > alpha:
        if p_for_near_sig:
            return "(" + value_name + f"={round(p, 3)})"
        else:
            return "n.s." if include_ns else ""
    elif mode == "p-values":
        return value_name + f"={round(p, 3)}"
    elif mode == "asterisks":
        return "*" * int(np.minimum(max_stars, np.floor(-np.log10(p))))


def p_list_to_disp_strings(
    p_list,
    alpha=0.05,
    include_ns=False,
    p_for_near_sig=True,
    mode="asterisks",
    value_name="p",
    max_stars=4,
):
    return [
        p_val_to_disp_string(
            p,
            alpha=alpha,
            include_ns=include_ns,
            p_for_near_sig=p_for_near_sig,
            mode=mode,
            max_stars=max_stars,
            value_name=value_name,
        )
        for p in p_list
    ]
