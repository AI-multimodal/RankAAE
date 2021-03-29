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
        print(features)
        self.assertArrayAlmostEqual(features,
                                    [[[5.1875850025644e-10, 9.149428725249412e-05, 217.27009087483182],
                                      [-1.0374962920650196e-10, -1.8292114017908488e-05, -12.78635920737967],
                                      [-3.1125233902748305e-10, -5.488758110804269e-05, -89.47184256811921]],
                                     [[2.8012220343046833e-10, 6.482524986317673e-05, 95.20603019472335],
                                      [-1.7403633745851342e-11, -6.0597735274769695e-06, -3.90522612457167],
                                      [-1.7735028949677561e-10, -3.764853731770096e-05, -36.97639252659645]]]
,
                                    3)            
