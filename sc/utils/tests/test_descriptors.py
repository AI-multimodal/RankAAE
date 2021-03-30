from unittest import TestCase

from matminer.featurizers.utils.grdf import Gaussian
from pymatgen import Structure, Lattice
from pymatgen.util.testing import PymatgenTest

from sc.utils.descriptors import AngularPDF


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

        # test transform,and featurize dataframe
        afs = AngularPDF(radial_bins=[fr1, fr2, fr3], angular_bins=[fa1, fa2, fa3])
        features = afs.transform(s_tuples)
        self.assertArrayAlmostEqual(features,
                                    [[[4.450242707991407e-19, 2.394524146670056e-12, 48.58854808988339], 
                                    [4.1500196821006496e-10, 7.317969397096165e-05, 59.29885267918986], 
                                    [4.450242707991393e-19, 2.3945241466700533e-12, 48.58854808988339]], 
                                    [[1.5509242704576662e-12, 9.085950337805756e-07, 10.39198745889294], 
                                    [1.2033044346926408e-14, 2.7838783476778295e-08, 27.863604246177356], 
                                    [2.456332302396376e-11, 5.0175487886531354e-06, 10.3693070139359]]],
                                    3)            
