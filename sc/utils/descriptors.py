import itertools
import numpy as np

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.grdf import Gaussian, Histogram

class AngularPDF(BaseFeaturizer):
    """
    Modified from matminer.featurizers.site.AngularFourierSeries 
    Compute the angular Fourier series (AFS), including both angular and radial info
    The AFS is the product of pairwise distance function (g_n, g_n') between two pairs
    of atoms (sharing the common central site) and the cosine of the angle
    between the two pairs. The AFS is a 2-dimensional feature (the axes are g_n,
    g_n').
    Examples of distance functionals are square functions, Gaussian, trig
    functions, and Bessel functions. An example for Gaussian:
        lambda d: exp( -(d - d_n)**2 ), where d_n is the coefficient for g_n
    See :func:`~matminer.featurizers.utils.grdf` for a full list of available binning functions.
    There are two preset conditions:
        gaussian: bin functions are gaussians
        histogram: bin functions are rectangular functions
    Features:
        AFS ([gn], [gn']) - Angular Fourier Series between binning functions (g1 and g2)
    Args:
        bins:   ([AbstractPairwise]) a list of binning functions that
                implement the AbstractPairwise base class
        cutoff: (float) maximum distance to look for neighbors. The
                 featurizer will run slowly for large distance cutoffs
                 because of the number of neighbor pairs scales as
                 the square of the number of neighbors
    """

    def __init__(self, radial_bins, angular_bins, radial_cutoff=10.0):
        self.radial_bins = radial_bins
        self.angular_bins = angular_bins
        self.radial_cutoff = radial_cutoff

    def featurize(self, struct, idx):
        """
        Get AFS of the input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            Flattened list of AFS values. the list order is:
                g_n g_n'
        """

        if not struct.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Generate list of neighbor position vectors (relative to central
        # atom) and distances from each central site as tuples
        sites = struct._sites
        central_site = sites[idx]
        neighbors_lst = struct.get_neighbors(central_site, self.radial_cutoff)
        neighbor_collection = [
            (neighbor[0].coords - central_site.coords, neighbor[1])
            for neighbor in neighbors_lst]

        # Generate exhaustive permutations of neighbor pairs around each
        # central site (order matters). Does not allow repeat elements (i.e.
        # there are two distinct sites in every permutation)
        neighbor_tuples = itertools.permutations(neighbor_collection, 2)

        # Generate cos(theta) between neighbor pairs for each central site.
        # Also, retain data on neighbor distances for each pair
        # process with matrix algebra, we really need the speed here
        data = np.array(list(neighbor_tuples))
        v1, v2 = np.vstack(data[:, 0, 0]), np.vstack(data[:, 1, 0])
        distances = data[:, :, 1]
        neighbor_pairs = np.concatenate([
            np.clip(np.einsum('ij,ij->i', v1, v2) /
                    np.linalg.norm(v1, axis=1) /
                    np.linalg.norm(v2, axis=1), -1.0, 1.0).reshape(-1, 1),
            distances], axis=1)

        # Generate distance functional matrix (g_n, g_n')

        # Compute AFS values for each element of the bin matrix
        # need to cast arrays as floats to use np.exp
        cos_angles, dist1, dist2 = neighbor_pairs[:, 0].astype(float),\
            neighbor_pairs[:, 1].astype(float),\
            neighbor_pairs[:, 2].astype(float)
        cos_angles = np.arccos(cos_angles)
        cos_angles = np.degrees(cos_angles)
        features = [[sum(rb(dist1) * rb(dist2) * ab(cos_angles))
                     for rb in self.radial_bins]
                    for ab in self.angular_bins]

        return features

    def feature_labels(self):
        return [[f'AFS {rb.name()} @ {ab.name()}' for rb in self.radial_bins]
                for ab in self.angular_bins]

    @staticmethod
    def from_preset(radial_preset, radial_width=0.5, radial_spacing=0.5, radial_start=0, radial_cutoff=10,
                    angular_preset="gaussian", angular_width=8.0, angular_spacing=8.0, angular_start=45, 
                    angular_cutoff=180):
        """
        Preset bin functions for this featurizer. Example use:
            >>> AFS = AngularFourierSeries.from_preset('gaussian')
            >>> AFS.featurize(struct, idx)
        Args:
            preset (str): shape of bin (either 'gaussian' or 'histogram')
            width (float): bin width. std dev for gaussian, width for histogram
            spacing (float): the spacing between bin centers
            cutoff (float): maximum distance to look for neighbors
        """

        # Generate radial bin functions
        if radial_preset == "gaussian":
            radial_bins = []
            for radial_center in np.arange(radial_start, radial_cutoff, radial_spacing):
                radial_bins.append(Gaussian(radial_width, radial_center))
        elif radial_preset == "histogram":
            radial_bins = []
            for start in np.arange(radial_start, radial_cutoff, radial_spacing):
                radial_bins.append(Histogram(start, radial_width))
        else:
            raise ValueError(f'Not a valid radial preset condition {radial_preset}.')

        # Generate angular bin functions
        if angular_preset == "gaussian":
            angular_bins = []
            for angular_center in np.arange(angular_start, angular_cutoff, angular_spacing):
                angular_bins.append(Gaussian(angular_width, angular_center))
        elif angular_preset == "histogram":
            angular_bins = []
            for start in np.arange(angular_start, angular_cutoff, angular_spacing):
                angular_bins.append(Histogram(start, angular_width))
        else:
            raise ValueError(f'Not a valid angular preset condition {angular_preset}.')

        return AngularPDF(radial_bins, angular_bins, radial_cutoff=radial_cutoff)

    def citations(self):
        return ['@article{PhysRevB.95.144110, title = {Representation of compo'
                'unds for machine-learning prediction of physical properties},'
                ' author = {Seko, Atsuto and Hayashi, Hiroyuki and Nakayama, '
                'Keita and Takahashi, Akira and Tanaka, Isao},'
                'journal = {Phys. Rev. B}, volume = {95}, issue = {14}, '
                'pages = {144110}, year = {2017}, publisher = {American Physic'
                'al Society}, doi = {10.1103/PhysRevB.95.144110}}']

    def implementors(self):
        return ["Maxwell Dylla", "Logan Williams", "Xiaohui Qu"]
