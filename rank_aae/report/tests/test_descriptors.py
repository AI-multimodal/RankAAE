from unittest import TestCase
import numpy as np

from matminer.featurizers.utils.grdf import Gaussian
from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from rank_aae.report.descriptors import AngularPDF, GeneralizedPartialRadialDistributionFunction


class TestAngularPDF(PymatgenTest):
    def setUp(self):
        self.sc = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            ["Al", ],
            [[0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False)
        self.cscl = Structure(
            Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
            ["Cl1-", "Cs1+"], [[0.45, 0.5, 0.5], [0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False)

    def test_afs(self):
        fr1 = Gaussian(1, 0)
        fr2 = Gaussian(1, 1)
        fr3 = Gaussian(1, 5)
        fa1 = Gaussian(5.0, 60)
        fa2 = Gaussian(5.0, 90.0)
        fa3 = Gaussian(5.0, 120.0)
        s_tuples = [(self.sc, 0), (self.cscl, 0)]

        radial_bin = [fr1, fr2, fr3]
        # test transform,and featurize dataframe
        afs = AngularPDF(radial_bins=[radial_bin],
                         angular_bins=[fa1, fa2, fa3])
        features = afs.transform(s_tuples)
        self.assertArrayAlmostEqual(features,
                                    [np.array([[3.39729162e-05, 2.83553809e-03, 7.24224653e+00],
                                               [1.00103478e-02, 2.06501706e-01,
                                                   1.02289003e+01],
                                               [3.39729162e-05, 2.83553809e-03, 7.24224653e+00]]),
                                     np.array([[2.07194688e-03, 6.11209555e-02, 6.13520637e+00],
                                               [6.99604354e-04, 2.86949946e-02,
                                                   9.00231999e+00],
                                               [3.13160638e-03, 7.72999765e-02, 6.10161855e+00]])],
                                    3)


class TestGeneralizedPartialRadialDistributionFunction(PymatgenTest):
    def setUp(self):
        self.sc = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            ["Al", ],
            [[0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False)
        self.cscl = Structure(
            Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
            ["Cl1-", "Cs1+"], [[0.45, 0.5, 0.5], [0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False)

    def test_prdf(self):
        prdf = GeneralizedPartialRadialDistributionFunction.from_preset(
            "gaussian")
        features = prdf.featurize(self.cscl, 0)
        expected_features = {"Cs": np.array([1.2335082e-04, 9.6556205e-03, 2.0891667e-01, 9.2267611e-01,
                                             7.1007483e-01, 1.2215943e-01, 4.2696887e-01, 8.0761593e-01,
                                             3.8646683e-01, 5.3590878e-01]),
                             "Cl": np.array([8.1303780e-07, 2.2786976e-04, 1.6962093e-02, 2.4544995e-01,
                                             6.0959051e-01, 5.3142959e-01, 6.3044031e-01, 4.1377699e-01,
                                             3.4712201e-01, 5.7155671e-01])}
        for k, v in expected_features.items():
            self.assertArrayAlmostEqual(features[k], v)

        prdf = GeneralizedPartialRadialDistributionFunction.from_preset(
            "histogram")
        features = prdf.featurize(self.cscl, 0)
        expected_features = {"Cs": np.array([0.       , 0.       , 0.       , 1.9244491, 0.       , 0.       ,
                                             0.8409994, 0.6319936, 0.       , 0.788243]),
                             "Cl": np.array([0.       , 0.       , 0.       , 0.       , 0.8754666, 1.1737025,
                                             0.       , 0.4213291, 0.2460989, 0.788243])}
        for k, v in expected_features.items():
            self.assertArrayAlmostEqual(features[k], v)
