#!/usr/bin/env python3

import numpy as np
# import os
import time

from pymatgen import MPRester
# from pymatgen.analysis.chemenv.coordination_environments\
#     .coordination_geometry_finder import LocalGeometryFinder

from sc.utils.logger import log
from sc.utils.timing import time_func, time_remaining


class Collection:
    """Container for data pulled directly from the Materials Project.
    """

    def __init__(
        self, api_key, patterns, max_atoms=np.inf, icsd_only=False,
        data_type='vasp'
    ):
        """
        Parameters
        ----------
        api_key : str
            The Materials Project API key for pulling data via pymatgen.
        patterns : list of str
            A list of various search patterns used for checking the Materials
            Project database. E.g. 'Ti-O', 'Ti-O-*' (wildcards are accepted).
        nmax_atoms : int
            The maximum number of atoms allowed in any given structure. Default
            is -1.
        icsd_only_ : bool
            Flag which if True only pulls data which is contained in the ICSD
            data. (Default is False).
        data_type : {'vasp', 'exp'}
            Data type one can pull. See here:
            https://pymatgen.org/_modules/pymatgen/ext/matproj.html
            Default is 'vasp'.
        """

        self.mpr = MPRester(api_key)
        mp_data = [
            self.mpr.get_data(search_pattern, data_type=data_type)
            for search_pattern in patterns
        ]

        def keep_condition(val):
            """Determines which data points to keep based on init
            parameters."""

            if val['nsites'] > max_atoms:
                return False
            if icsd_only:
                if val['icsd_id'] or val['icsd_ids']:
                    return False
            return True

        self.raw_mp_data = {
            val['material_id']: val for sublist in mp_data for val in sublist
            if keep_condition(val)
        }

        log.info(f"Patterns: {patterns}")
        log.info(f"Max atoms/sample: {max_atoms}")
        log.info(f"ICSD only: {icsd_only}")
        log.info(f"Data type: {data_type}")
        log.info(f"Found {len(self.raw_mp_data)} structures.")

        self.patterns = patterns
        self.max_atoms = max_atoms
        self.icsd_only = icsd_only
        self.data_type = data_type
        self.processed_mp_data = None

    @time_func(log)
    def process_raw_mp_data(self, log_every=100):
        """Runs the raw Materials Project data through the processing pipeline.
        This amounts to pulling down feff xas data into a dictionary.

        Parameters
        ----------
        log_every : int
        """

        self.processed_mp_data = dict()
        cc = 0

        t0 = time.time()
        for mpid, __ in self.raw_mp_data.items():
            self.processed_mp_data[mpid] = \
                self.mpr.get_data(mpid, data_type='feff', prop='xas')
            cc += 1
            if cc % log_every == 0:
                remaining = time_remaining(
                    time.time() - t0, cc / len(self.raw_mp_data) * 100.0
                )
                log.info(f"ETA {(remaining/60.0):.02f} m\t {mpid} done")

    def download_CONTCAR(self, path):
        """Creates a directory structure for storing the CONTCAR files and
        saves to disk.

        Parameters
        ----------
        path : str
            The path to the location to create and save the CONTCAR files.
        """

        raise NotImplementedError
