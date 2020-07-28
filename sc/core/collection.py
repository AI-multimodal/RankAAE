#!/usr/bin/env python3

import os

from pymatgen import MPRester

from sc.utils.logger import log


class Collection:
    """Container for data pulled directly from the Materials Project.
    """

    def __init__(
        self, api_key, patterns, max_atoms=200, icsd_only=False,
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
            is 200.
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

        self.raw_mp_data = [
            val for sublist in mp_data for val in sublist
            if keep_condition(val)
        ]

        log.info(f"Patterns: {patterns}")
        log.info(f"Max atoms/sample: {max_atoms}")
        log.info(f"ICSD only: {icsd_only}")
        log.info(f"Data type: {data_type}")
        log.info(f"Found {len(mp_data)} structures.")

        self.patterns = patterns
        self.max_atoms = max_atoms
        self.icsd_only = icsd_only
        self.data_type = data_type

    def process_structures(self):
        """Runs the raw Materials Project data through the processing pipeline.
        This includes getting structural and spectral information."""

        raise NotImplementedError

    def download_CONTCAR(self, path):
        """Creates a directory structure for storing the CONTCAR files and
        saves to disk.

        Parameters
        ----------
        path : str
            The path to the location to create and save the CONTCAR files.
        """

        raise NotImplementedError
