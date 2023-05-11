# -*- coding: utf-8 -*-

"""Top-level package for astro_custom."""

__author__ = """J. Michael Burgess"""
__email__ = 'jburgess@mpe.mpg.de'


from .utils.configuration import astro_custom_config, show_configuration
from .utils.logging import (
    update_logging_level,
    activate_warnings,
    silence_warnings,
)


from .tbabs_cut import TbAbsCut

from .contour import contour_plot
