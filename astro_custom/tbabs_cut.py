import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import astropy.units as astropy_units
import numba as nb
import numpy as np
from astropy.io import fits
from interpolation import interp

from astromodels.functions.function import Function1D, FunctionMeta
from astromodels.utils import _get_data_file_path
from astromodels.utils.configuration import astromodels_config
from astromodels.utils.logging import setup_logger
from astromodels.functions.functions_1D.absorption import tbabs

log = setup_logger(__name__)


@nb.vectorize
def _exp(x):
    return math.exp(x)


@nb.njit(fastmath=True)
def _numba_eval(nh, xsect_interp):

    return _exp(-nh * xsect_interp)


class TbAbsCut(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Photometric absorption (Tbabs implementation), f(E) = exp(- NH * sigma(E))
        contributed by Dominique Eckert
    parameters :
        NH :
            desc : absorbing column density in units of 1e22 particles per cm^2
            initial value : 1.0
            is_normalization : True
            transformation : log10
            min : 1e-4
            max : 1e4
            delta : 0.1

        redshift :
            desc : the redshift of the source
            initial value : 0.
            is_normalization : False
            min : 0
            max : 15
            delta : 0.1
            fix: True

        low_cutoff:
            desc: energy where things go back to 1
            initial value: 1e-2
            fix: True




    properties:
         abundance_table:
            desc: the abundance table for the model
            initial value: WILM
            allowed values:
            - WILM
            - AG89
            - ASPL
            function: _init_xsect

    """

    def _setup(self):

        # self.init_xsect(self.abundance_table)

        self._fixed_units = (
            astropy_units.keV,
            astropy_units.dimensionless_unscaled,
        )

    def _set_units(self, x_unit, y_unit):

        self.NH.unit = astropy_units.cm ** (-2)
        self.redshift.unit = astropy_units.dimensionless_unscaled
        self.low_cutoff.unit = x_unit

    def _init_xsect(self):
        """
        Set the abundance table

        :param abund_table: "WILM", "ASPL", "AG89"
        :returns:
        :rtype:

        """

        tbabs.set_table(self.abundance_table.value)

        self.xsect_ene, self.xsect_val = tbabs.xsect_table

        log.debug(f"updated the TbAbs table to {self.abundance_table.value}")

    @property
    def abundance_table_info(self):
        print(tbabs.info)

    def evaluate(self, x, NH, redshift, low_cutoff):

        if isinstance(x, astropy_units.Quantity):

            _unit = astropy_units.cm**2
            _y_unit = astropy_units.dimensionless_unscaled
            _x = x.value
            _redshift = redshift.value
        else:

            _unit = 1.0
            _y_unit = 1.0
            _redshift = redshift
            _x = x

        xsect_interp = interp(
            self.xsect_ene, self.xsect_val, _x * (1 + _redshift)
        )

        spec = _numba_eval(NH, xsect_interp)

        spec[x < low_cutoff] = 1.0

        return spec * _y_unit
