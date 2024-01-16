"""This module contains the objects used to visualize the data for the project
"""
# --------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------

from IPython.display import HTML, display

# Plotting libraries
from plotly import graph_objects as go


# --------------------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------------------
def generate_plotly_transparent_template(
    color_sequence: list,
    title_font_color: str,
    axis_font_color: str,
    axis_tick_color: str,
    axis_grid_color: str,
    legend_font_color: str,
    plot_width: int,
    plot_height: int,
    plot_bg_color: str = "rgba(0,0,0,0)",
    paper_bg_color: str = "rgba(0,0,0,0)",
) -> go.layout.Template:
    """This function generates a Plotly Template with custom colors and
    stable plot shape.

    Args:
        color_sequence (list): list of colors
        title_font_color (str): title font color
        axis_font_color (str): axis font color
        axis_tick_color (str): axis tick color
        axis_grid_color (str): axis line-grid color
        plot_width (int): plot width
        plot_height (int): plot height
        plot_bg_color (str, optional): this sets the plot background.
            Defaults to 'rgba(0,0,0,0)'.
        paper_bg_color (str, optional): this sets the paper background.
            Defaults to 'rgba(0,0,0,0)'.

    Returns:
        go.layout.Template: Plotly template with custom colors
    """
    template = go.layout.Template(
        layout=go.Layout(
            plot_bgcolor=plot_bg_color,  # Transparent plot background
            paper_bgcolor=paper_bg_color,  # Transparent paper background
            xaxis=dict(
                title_font=dict(color=axis_font_color),  # Set x-axis title color
                tickfont=dict(color=axis_tick_color),  # Set x-axis tick color
                gridcolor=axis_grid_color,  # Set x-axis grid line color
            ),
            yaxis=dict(
                title_font=dict(color=axis_font_color),  # Set y-axis title color
                tickfont=dict(color=axis_tick_color),  # Set y-axis tick color
                gridcolor=axis_grid_color,  # Set y-axis grid line color
            ),
            title=dict(font=dict(color=title_font_color)),  # Set plot title color
            colorway=color_sequence,  # Set the custom color sequence
            legend=dict(font=dict(color=legend_font_color)),
            width=plot_width,
            height=plot_height,
        )
    )

    return template


def visualize_hex_colors(hex_colors):
    """
    Generate a visualization of a list of hex colors.

    Parameters:
        hex_colors (List[str]): A list of hex color codes.

    Returns:
        None
    """
    # Create a string of HTML to display the colors
    html_str = '<div style="display: flex;">'
    for color in hex_colors:
        html_str += (
            f'<div style="width: 30px; height: 30px; background: {color};'
            f'margin: 2px;"></div>'
        )
    html_str += "</div>"

    # Display the HTML in the Jupyter notebook
    display(HTML(html_str))


# --------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# --------------------------------------------------------------------------------------
FONT_COLOR = "#F2F3F4"
AXIS_COLOR = "#5D6D7E"
CYBERPUNK_COLOR_SEQUENCE = [
    "#39FF14",
    "#FF1493",
    "#00FFFF",
    "#9400D3",
    "#FFFF00",
    "#FF4500",
    "#32CD32",
    "#87CEEB",
    "#FF00FF",
    "#FF0000",
]
WIDTH = 1400
HEIGHT = 700
