from unittest import TestCase
import numpy as np

from matminer.featurizers.utils.grdf import Gaussian
from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from sc.utils.descriptors import AngularPDF, GeneralizedPartialRadialDistributionFunction


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
        afs = AngularPDF(radial_bins=[radial_bin], angular_bins=[fa1, fa2, fa3])
        features = afs.transform(s_tuples)
        self.assertArrayAlmostEqual(features,
                                    [np.array([[3.39729162e-05, 2.83553809e-03, 7.24224653e+00],
                                               [1.00103478e-02, 2.06501706e-01, 1.02289003e+01],
                                               [3.39729162e-05, 2.83553809e-03, 7.24224653e+00]]), 
                                     np.array([[2.07194688e-03, 6.11209555e-02, 6.13520637e+00],
                                               [6.99604354e-04, 2.86949946e-02, 9.00231999e+00],
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
        print('hhh')
        prdf = GeneralizedPartialRadialDistributionFunction.from_preset("gaussian")
        features = prdf.featurize(self.cscl, 0)
        expected_features = {"Species Cs+": np.array([3.30853148e-06, 2.58984283e-04, 5.60358944e-03, 2.47481359e-02, 1.90457175e-02, 3.27657575e-03, 
                                                      1.14522135e-02, 2.16619771e-02, 1.03658626e-02, 1.43742135e-02]), 
                             "Species Cl-": np.array([2.18074034e-08, 6.11195171e-06, 4.54959413e-04, 6.58348968e-03, 1.63505142e-02, 1.42540722e-02, 
                                                      1.69097501e-02, 1.10983790e-02, 9.31055080e-03, 1.53303668e-02])}
        self.assertDictsAlmostEqual(features, expected_features)
