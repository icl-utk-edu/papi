#!/usr/bin/env python3

# Python standard library packages
import argparse
import json
import subprocess

# Plotting packages
import matplotlib as mpl 
import matplotlib.pyplot as plt

# Misc. packages
import numpy as np

def setup_args() -> argparse.ArgumentParser:
    """Setup the command line interface.

    :returns: An instance of argparse.ArgumentParser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--starting-release", help = "The release you want to begin plotting from. Default is 0 and correponds to the first PAPI GitHub release.")
    parser.add_argument("--figsize", help = "The size of the generated figures. Must be in the format width_size,height_size.")
    parser.add_argument("--color", help = "The color to be used in bars (barplot) or points (lineplot + scatterpot). You can pass a colormap or list of colors. In the case of a list of colors, they must be comma separated and the number provided must match the number of releases you wish to plot.")
    parser.add_argument("--fontsize", help = "Fontsizes for the plots x and y ticks/labels and title.")
    parser.add_argument("--barplot-kwargs", help = "Keyword arguments for the generated barplot.")
    parser.add_argument("--lineplot-kwargs", help = "Keyword arguments for the generated lineplot.")
    parser.add_argument("--scatterplot-kwargs", help = "Keyword arguments for the generated scatterplot.")
    parser.add_argument("--filename-barplot-download-count", help = "The filename for the generated barplot.")
    parser.add_argument("--filename-lineplot-download-count", help = "The filename for the generated lineplot.")

    return parser

def get_papi_release_download_count(starting_release_arg: int) -> tuple[int, list[str], list[int]]:
    """For each PAPI release on GitHub get the download count.

    :returns: A tuple containing number of releases, name of releases, and the download count per release.
    :rtype: tuple
    """
    repo_owner = "icl-utk-edu"
    repo = "papi"

    papi_release_info_json = subprocess.run(f"gh api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' /repos/{repo_owner}/{repo}/releases", shell = True, capture_output = True, text = True)
    papi_release_info_python = json.loads(papi_release_info_json.stdout)
    papi_release_info_python.reverse()

    names_of_releases = []
    number_of_downloads_per_release = []
    for release_entry in papi_release_info_python[starting_release_arg:]:
        names_of_releases.append(release_entry["name"])
        number_of_downloads_per_release.append(release_entry["assets"][0]["download_count"])

    return len(names_of_releases), names_of_releases, number_of_downloads_per_release

def plot_download_count_for_papi_gh_releases(names_of_gh_papi_releases_arg: list, number_of_gh_download_counts_per_papi_release_arg: list,  parsed_command_line_args: tuple) -> None:
    """Plot the PAPI GitHub release download count as a barplot and lineplot.

    :param names_of_gh_papi_releases_arg: A list of PAPI GitHub release names.
    :type names_of_gh_papi_releases_arg: list
    :param number_of_gh_download_counts_per_papi_release_arg: A list of download counts per PAPI GitHub release.
    :type number_of_gh_download_counts_per_papi_release_arg: list
    :param parsed_command_line_args: A tuple containing the parsed arguments from the command line interface.
    :type parsed_command_line_args: tuple
    """
    # Unpack the tuple of arguments that were created in parse_command_line_args
    plots_figsize, plots_color, plots_fontsize, plots_barplot_kwargs, plots_lineplot_kwargs, plots_scatterplot_kwargs, plots_filenames = parsed_command_line_args

    plots_to_generate = ["bar", "line"]
    for plot,filename in zip(plots_to_generate, plots_filenames):
        fig, ax = plt.subplots(figsize = plots_figsize)
        # Showcase the PAPI GitHub download count via a barplot
        if plot == "bar":
            bars = ax.bar(x = names_of_gh_papi_releases_arg, height = number_of_gh_download_counts_per_papi_release_arg, color = plots_color, **plots_barplot_kwargs)
            ax.bar_label(bars, padding = 3, fontsize = plots_fontsize)
        # Showcase the PAPI GitHub download count via a lineplot
        elif plot == "line":
            ax.plot(names_of_gh_papi_releases_arg, number_of_gh_download_counts_per_papi_release_arg, zorder = 0, **plots_lineplot_kwargs)
            ax.scatter(x = names_of_gh_papi_releases_arg, y = number_of_gh_download_counts_per_papi_release_arg, zorder = 1, color = plots_color, **plots_scatterplot_kwargs)
            # Add annotations
            for name, count in zip(names_of_gh_papi_releases_arg, number_of_gh_download_counts_per_papi_release_arg):
                ax.annotate(f"{count}", xy=(name, count + 70), ha = "center", fontsize = plots_fontsize)
        # Plot option has yet to be implemented
        else:
            raise NotImplementedError

        # Handle the figures title
        plots_title = "The Number of Downloads per PAPI Release via GitHub"
        ax.set_title(plots_title, fontsize = plots_fontsize)

        # Handle the figures y-axis
        plots_ylabel = "Number of Downloads"
        yaxis_stepsize = 300
        yaxis_current_max = max(number_of_gh_download_counts_per_papi_release_arg)
        ## The y-axis max is updated to be + yaxis_stepsize such that
        ## a value is placed at the top left corner of the plot
        yaxis_updated_max = yaxis_current_max + yaxis_stepsize
        ax.set_yticks(np.arange(0, yaxis_updated_max, yaxis_stepsize))
        ax.tick_params(axis = "y", labelsize = plots_fontsize)
        ax.set_ylabel(plots_ylabel, fontsize = plots_fontsize)

        # Handle the figures x-axis
        plots_xlabel = "PAPI Releases"
        ax.set_xlabel(plots_xlabel, fontsize = plots_fontsize)
        ax.tick_params(axis = "x", labelsize = plots_fontsize)

        # Save the figure
        fig.tight_layout()
        fig.savefig(f"{filename}.svg", format='svg', dpi=1200)

def parse_command_line_args(cmd_line_args: argparse.Namespace, number_of_gh_papi_releases_arg: int):
    """Parse the command line interface args.

    :param cmd_line_args: Namespace containing data for our command line arguments.
    :type cmd_line_args: argparse.Namespace
    :param number_of_gh_papi_releases_arg: Number of GitHub PAPI releases to plot.
    :type int
    :returns: A tuple containing the parsed arguments which will be used to generate the barplot and lineplot.
    :rtype: tuple
    """
    # Group the filenames into a list
    filenames = [cmd_line_args.filename_barplot_download_count,
                 cmd_line_args.filename_lineplot_download_count]

    # Determine the figure size
    seperator = ","
    try:
        width, height = cmd_line_args.figsize.split(seperator)
    except ValueError as e:
        e.add_note("The argument to adjust figure size is not formatted properly. Format must be width_size,height_size.")
        raise
    figsize = (int(width), int(height))

    # Determine the colors for the bars (barplot) or points (lineplot + scatterplot)
    color = None
    if "," in cmd_line_args.color:
        seperator = ","
        color = cmd_line_args.color.split(seperator)
        if len(colors) != number_of_gh_papi_releases_arg:
            raise ValueError(f"A total of {len(colors)} colors were provided to --colors, but {number_of_papi_releases_on_gh} releases are being plotted.")
    else:
        try:
            cmp = mpl.colormaps[cmd_line_args.color]
        except KeyError as e:
            e.add_note(f"The colormap {cmd_line_args.colors} provided to --colors is not actually a colormap.")
            raise
        color = cmp(np.linspace(0, 1, number_of_gh_papi_releases_arg))

    # Convert the argument JSON format for --barplot-kwargs
    # to a Python dictionary
    barplot_kwargs = json.loads(cmd_line_args.barplot_kwargs)

    # Convert the argument JSON format for --lineplot-kwargs
    # to a Python dictionary
    lineplot_kwargs = json.loads(cmd_line_args.lineplot_kwargs)

    # Convert the argument JSON format for --scatterplot-kwargs
    # to a Python dictionary
    scatterplot_kwargs = json.loads(cmd_line_args.scatterplot_kwargs)

    return figsize, color, cmd_line_args.fontsize, barplot_kwargs, lineplot_kwargs, scatterplot_kwargs, filenames
     
if __name__ == "__main__":
    parser = setup_args()
    args = parser.parse_args()

    number_of_gh_papi_releases, names_of_gh_papi_releases, number_of_gh_download_counts_per_papi_release = get_papi_release_download_count(int(args.starting_release))

    command_line_arguments_parsed = parse_command_line_args(args, number_of_gh_papi_releases)

    plot_download_count_for_papi_gh_releases(names_of_gh_papi_releases, number_of_gh_download_counts_per_papi_release, command_line_arguments_parsed)
