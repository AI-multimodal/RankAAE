from unittest import TestCase

from matminer.featurizers.utils.grdf import Gaussian
from pymatgen import Structure, Lattice
from pymatgen.util.testing import PymatgenTest

from sc.utils.descriptors import AngularFourierSeries


class TestAngularFourierSeries(PymatgenTest):
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
        f1 = Gaussian(1, 0)
        f2 = Gaussian(1, 1)
        f3 = Gaussian(1, 5)
        s_tuples = [(self.sc, 0), (self.cscl, 0)]

        # test transform,and featurize dataframe
        afs = AngularFourierSeries(bins=[f1, f2, f3])
        features = afs.transform(s_tuples)
        self.assertArrayAlmostEqual(features,
                                    [[-1.0374e-10, -4.3563e-08, -2.7914e-06,
                                      -4.3563e-08, -1.8292e-05, -0.0011,
                                      -2.7914e-06, -0.0011, -12.7863],
                                     [-1.7403e-11, -1.0886e-08, -3.5985e-06,
                                      -1.0886e-08, -6.0597e-06, -0.0016,
                                      -3.5985e-06, -0.0016, -3.9052]],
                                    3)            
