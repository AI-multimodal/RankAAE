import itertools
import numpy as np
import math

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.grdf import Gaussian, Histogram
from pymatgen.core.structure import Structure

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
        radial_bins: a list containing either one (for single width) or two (for double width)
                     [AbstractPairwise] lists (see below for more details)
        angular_bins:   ([AbstractPairwise]) a list of binning functions that
                implement the AbstractPairwise base class
        cutoff: (float) maximum distance to look for neighbors. The
                 featurizer will run slowly for large distance cutoffs
                 because of the number of neighbor pairs scales as
                 the square of the number of neighbors
    """

    def __init__(self, radial_bins, angular_bins, radial_cutoff=10.0):
        
        # duplicate radial_bins if single width is used.
        self.radial_bins = radial_bins
        if len(radial_bins) == 1:
            self.radial_bins = self.radial_bins * 2
        
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

        # Calcuate features. Note that if single width is used, rb1=rb2.
        features = [[sum(np.sqrt(rb1(dist1) * rb2(dist2)) * ab(cos_angles))
                    for (rb1,rb2) in zip(self.radial_bins[0],self.radial_bins[1])]
                    for ab in self.angular_bins]
        
        features = np.array(features)
        features = np.sqrt(features)
        return features

    def feature_labels(self):
        return [[f'AFS {rb1.name()} and {rb2.name()} @ {ab.name()}' 
                for (rb1,rb2) in zip(self.radial_bins[0],self.radial_bins[1])]
                for ab in self.angular_bins]

    @staticmethod
    def from_preset(radial_preset, radial_width=[0.5,0.5], radial_spacing=0.5, radial_start=0, radial_cutoff=0,
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
        radial_bins = []
        for i in range(len(radial_width)):
            if radial_preset == "gaussian":
                radial_bin = [Gaussian(radial_width[i], radial_center)
                              for radial_center in np.arange(radial_start, radial_cutoff, radial_spacing)]
                radial_bins.append(radial_bin)
            elif radial_preset == "histogram":
                radial_bin = [Histogram(start, radial_width[i])
                              for start in np.arange(radial_start, radial_cutoff, radial_spacing)]
                radial_bins.append(radial_bin)
            else:
                raise ValueError(f'Not a valid radial preset condition {radial_preset}.')

        # Generate angular bin functions
        angular_bins = []
        if angular_preset == "gaussian":
            angular_bin = [Gaussian(angular_width, angular_center)
                           for angular_center in np.arange(angular_start, angular_cutoff, angular_spacing)]
            angular_bins = angular_bin
        elif angular_preset == "histogram":
            angular_bin = [Histogram(start, angular_width)
                           for start in np.arange(angular_start, angular_cutoff, angular_spacing)]
            angular_bins = angular_bin
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


class GeneralizedPartialRadialDistributionFunction(BaseFeaturizer):
    """
    Compute the general radial distribution function (GRDF) for a site.
    The GRDF is a radial measure of crystal order around a site. There are two
    featurizing modes:
    1. GRDF: (recommended) - n_bins length vector
        In GRDF mode, The GRDF is computed by considering all sites around a
        central site (i.e., no sites are omitted when computing the GRDF). The
        features output from this mode will be vectors with length n_bins.
    2. pairwise GRDF: (advanced users) - n_bins x n_sites matrix
        In this mode, GRDFs are are still computed around a central site, but
        only one other site (and their translational equivalents) are used to
        compute a GRDF (e.g. site 1 with site 2 and the translational
        equivalents of site 2). This results in a a n_sites x n_bins matrix of
        features. Requires `fit` for determining the max number of sites for
    3. Element Partial GRDF:
        In this mode, GRDFs are are still computed around a central site, but
        only one other element to compute GRDF.
    The GRDF is a generalization of the partial radial distribution function
    (PRDF). In contrast with the PRDF, the bins of the GRDF are not mutually-
    exclusive and need not carry a constant weight of 1. The PRDF is a case of
    the GRDF when the bins are rectangular functions. Examples of other
    functions to use with the GRDF are Gaussian, trig, and Bessel functions.
    See :func:`~matminer.featurizers.utils.grdf` for a full list of available binning functions.
    There are two preset conditions:
        gaussian: bin functions are gaussians
        histogram: bin functions are rectangular functions
    Args:
        bins:   ([AbstractPairwise]) List of pairwise binning functions. Each of these functions
            must implement the AbstractPairwise class.
        cutoff: (float) maximum distance to look for neighbors
        mode:   (str) the featurizing mode. supported options are:
                    'GRDF', 'pairwise_GRDF' and "element_partial_GRDF"
    """

    def __init__(self, bins, cutoff=20.0, mode="element_partial_GRDF"):
        self.bins = bins
        self.cutoff = cutoff

        if mode not in ["GRDF", "pairwise_GRDF", "element_partial_GRDF"]:
            raise AttributeError("{} is not a valid GRDF mode. try " '"GRDF", "pairwise_GRDF" or "element_partial_GRDF"'.format(mode))
        else:
            self.mode = mode

        self.fit_labels = None

    def fit(self, X, y=None, **fit_kwargs):
        """
        Determine the maximum number of sites in X to assign correct feature
        labels
        Args:
            X - [list of tuples], training data
                tuple values should be (struc, idx)
        Returns:
            self
        """

        max_sites = max([len(X[i][0]._sites) for i in range(len(X))])
        self.fit_labels = ["site2 {} {}".format(i, bin.name()) for bin in self.bins for i in range(max_sites)]
        return self

    def featurize(self, struct: Structure, idx):
        """
        Get GRDF of the input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            Flattened list of GRDF values. For each run mode the list order is:
                GRDF:          bin#
                pairwise GRDF: site2# bin#
                element partial GRDF: element2# bin#
            The site2# corresponds to a pymatgen site index and bin#
            corresponds to one of the bin functions
        """

        if not struct.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get list of neighbors by site
        # Indexing is [site#][neighbor#][pymatgen Site, distance, site index]
        sites = struct._sites
        norm_factor = struct.volume / len(sites)
        central_site = sites[idx]
        neighbors_lst = struct.get_neighbors(central_site, self.cutoff, include_index=True)
        sites = range(0, len(sites))
        self.elements = tuple([sp.symbol for sp in set(struct.species)])

        # Generate lists of pairwise distances according to run mode
        if self.mode == "GRDF":
            # Make a single distance collection
            distance_collection = [[neighbor[1] for neighbor in neighbors_lst]]
        elif self.mode == "pairwise_GRDF":
            # Make pairwise distance collections for pairwise GRDF
            distance_collection = [
                [neighbor[1] for neighbor in neighbors_lst if neighbor[2] == site_idx] for site_idx in sites
            ]
        else:
            assert self.mode == "element_partial_GRDF"
            distance_collection = [
                [neighbor[1] for neighbor in neighbors_lst if neighbor.specie.symbol == ele] for ele in self.elements
            ]


        # compute bin counts for each list of pairwise distances
        bin_counts = []
        for values in distance_collection:
            bin_counts.append([sum(bin(values)) for bin in self.bins])

        # Compute "volume" of each bin to normalize GRDFs
        volumes = [bin.volume(self.cutoff) for bin in self.bins]

        # normalize the bin counts by the bin volume to compute features
        if self.mode in ["GRDF", "pairwise_GRDF"]:
            features = []
            for values in bin_counts:
                features.extend(np.array(values) / np.array(volumes))
        else:
            assert self.mode == "element_partial_GRDF"
            features = dict()
            for ele, values in zip(self.elements, bin_counts):
                features[ele] = norm_factor * np.array(values) / np.array(volumes)

        return features

    def feature_labels(self):
        if self.mode == "GRDF":
            return [bin.name() for bin in self.bins]
        elif self.mode == "pairwise_GRDF":
            if self.fit_labels:
                return self.fit_labels
            else:
                raise AttributeError("the fit method must be called first, to " "determine the correct feature labels.")
        else:
            assert self.mode == "element_partial_GRDF"
            return [f"With {ele} {bin.name()}" for bin in self.bins for ele in self.elements]

    @staticmethod
    def from_preset(preset, width=1.0, spacing=1.0, cutoff=10, mode="element_partial_GRDF"):
        """
        Preset bin functions for this featurizer. Example use:
            >>> GRDF = GeneralizedRadialDistributionFunction.from_preset('gaussian')
            >>> GRDF.featurize(struct, idx)
        Args:
            preset (str): shape of bin (either 'gaussian' or 'histogram')
            width (float): bin width. std dev for gaussian, width for histogram
            spacing (float): the spacing between bin centers
            cutoff (float): maximum distance to look for neighbors
            mode (str): featurizing mode. either 'GRDF' or 'pairwise_GRDF'
        """

        # Generate bin functions
        if preset == "gaussian":
            bins = []
            for center in np.arange(0.0, cutoff, spacing):
                bins.append(Gaussian(width, center))
        elif preset == "histogram":
            bins = []
            for start in np.arange(0, cutoff, spacing):
                bins.append(Histogram(start, width))
        else:
            raise ValueError("Not a valid preset condition.")
        return GeneralizedPartialRadialDistributionFunction(bins, cutoff=cutoff, mode=mode)

    def citations(self):
        return [
            "@article{PhysRevB.95.144110, title = {Representation of compo"
            "unds for machine-learning prediction of physical properties},"
            " author = {Seko, Atsuto and Hayashi, Hiroyuki and Nakayama, "
            "Keita and Takahashi, Akira and Tanaka, Isao},"
            "journal = {Phys. Rev. B}, volume = {95}, issue = {14}, "
            "pages = {144110}, year = {2017}, publisher = {American Physic"
            "al Society}, doi = {10.1103/PhysRevB.95.144110}}"
        ]

    def implementors(self):
        return ["Maxwell Dylla", "Saurabh Bajaj", "Logan Williams", "Xiaohui Qu"]