#!/usr/bin/env python3

import numpy as np
# import os
import sys
import time

from pymatgen import MPRester
from pymatgen.analysis.chemenv.coordination_environments\
    .coordination_geometry_finder import LocalGeometryFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from rank_aae.legacy.symmetry import get_cesym

from rank_aae.legacy.logger import log as default_logger
from rank_aae.legacy.timing import time_func, time_remaining


# Initializng the LGF produces a ridiculous amount of logging output for
# no good reason. Pipe it to the trash
TRASH = ".trash_COLLECTION"
save_stdout = sys.stdout
sys.stdout = open(TRASH, 'w')
LGF = LocalGeometryFinder()
sys.stdout = save_stdout


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

        default_logger.info(f"Patterns: {patterns}")
        default_logger.info(f"Max atoms/sample: {max_atoms}")
        default_logger.info(f"ICSD only: {icsd_only}")
        default_logger.info(f"Data type: {data_type}")
        default_logger.info(f"Found {len(self.raw_mp_data)} structures.")

        self.patterns = patterns
        self.max_atoms = max_atoms
        self.icsd_only = icsd_only
        self.data_type = data_type
        self.processed_mp_data = None

    @time_func(default_logger)
    def process_raw_mp_data(self, log_every=100, debug=-1):
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
                default_logger.info(
                    f"ETA {(remaining/60.0):.02f} m\t {mpid} done"
                )

            if debug > -1 and cc > debug:
                break

    def get_symmetry_information(self, abs_gt):
        """

        Parameters
        ----------
        abs_gt : str
            The "ground truth" confirmation of the absorbing atom provided
            by the user, e.g. 'Ti'.
        """

        self.symmetry_info = dict()
        cc = 0

        for mpid, data in self.processed_mp_data.items():
            self.symmetry_info[mpid] = dict()

            default_logger.info(f"{cc:08} Processing {mpid}")

            scratch = []
            for xas_doc in data[0]['xas']:
                data_abs_species = xas_doc['structure'] \
                    .species[xas_doc['absorbing_atom']].name
                data_structure = xas_doc['structure']
                data_absorption_atom = xas_doc['absorbing_atom']

                if data_abs_species != abs_gt:
                    continue

                # Get the spectrum
                spectrum_x, spectrum_y = xas_doc['spectrum']
                self.symmetry_info[mpid]['abs_idx'] = xas_doc['absorbing_atom']
                self.symmetry_info[mpid]['spectrum_x'] = spectrum_x
                self.symmetry_info[mpid]['spectrum_y'] = spectrum_y

                finder = SpacegroupAnalyzer(data_structure)
                structure = finder.get_symmetrized_structure()
                sites = structure.equivalent_sites
                indices = structure.equivalent_indices

                # find the particular atom index
                for ii, mm in enumerate(indices):
                    if data_absorption_atom in mm and ii not in scratch:
                        multiplicity = len(mm)
                        scratch.append(ii)
                        break

                # Save important metadata
                self.symmetry_info[mpid]['multiplicity'] = multiplicity
                self.symmetry_info[mpid]['equivalent_sites'] = sites
                self.symmetry_info[mpid]['abs_atom'] = data_absorption_atom

                # The good stuff, the symmetry information:
                # get rid of any indices that do not correspond to the absorber
                indices = [
                    index_ for index_ in indices
                    if structure[index_[0]].species_string == abs_gt
                ]

                cesyms = []
                csms = []
                equiv = []

                for equivalent_indices in indices:
                    cesym = get_cesym(LGF, structure, equivalent_indices)
                    cesyms.append(cesym[0])
                    csms.append(cesym[1])
                    equiv.append(equivalent_indices)

                np.testing.assert_equal(len(cesyms), len(csms))
                np.testing.assert_equal(len(equiv), len(csms))

                self.symmetry_info[mpid]['cesyms'] = cesyms
                self.symmetry_info[mpid]['csms'] = csms
                self.symmetry_info[mpid]['equiv'] = equiv

    def download_CONTCAR(self, path):
        """Creates a directory structure for storing the CONTCAR files and
        saves to disk.

        Parameters
        ----------
        path : str
            The path to the location to create and save the CONTCAR files.
        """

        raise NotImplementedError
