
import sys
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.interpolate import UnivariateSpline
sys.path.append("/home/zliang/Documents/XANES_descriptors/")
from pyfitit import curveFitting, descriptor
from scipy.signal import find_peaks


class SpecDescriptors():
    
    def __init__(self, grid, spec):
        self.grid = grid
        self.spec = spec
        self.spline = None
        self.arctan = None
        self._energy = None
        self.update()
    
    @property
    def descriptors(self):
        descriptors = {
            "edge": {
                "position": None, "slope": None, "intensity": None
            },
            "main_peak": {
                "position": None, "intensity": None, "curvature": None
            },
            "pit": {
                "position": None, "intensity": None, "curvature": None
            },
            "last_peak": {
                "position": None, "intensity": None, "curvature": None
            },
            "sec_peak": {
                "position": None, "intensity": None, "curvature": None
            },
            "pre_peak": {
                "positioin": None, "intensity": None, "curvature": None
            },
            "other": {
                "main_last_separation": None, "main_pit_separation": None,
                "pit_last_spread": None, "pit_last_separation": None,
                "fluctuation": None
            },
        }
        return descriptors
    
    @classmethod
    def from_spline(cls, grid, spec, * , fine_grid, k=5, s=0.01):
        spl_fun = UnivariateSpline(grid, spec, k=k, s=s)
        spec_des_object =  SpecDescriptors(fine_grid, spl_fun(fine_grid))
        spec_des_object.spline = spl_fun
        return spec_des_object
    
    def update(self):
        for key, value in self.descriptors.items():
            self.__dict__[key] = value
    
    def find_edge(self):
        """
        find edge position and slope for the spectra.
        """ 
        result = curveFitting.findEfermiByArcTan(self.grid, self.spec)
        self.arctan = result[1]
        pos_index = np.argmin(abs(self.grid - result[0]['x0']))
        self.edge["position"] = self.grid[pos_index]
        self.edge["intensity"] = self.spec[pos_index]

        spl_d1 = self.spline.derivative(n=1)
        self.edge["slope"] = float(spl_d1(self.grid[pos_index]))
    
    
    def find_main_peak(self, window=1, left=None, right=None, width=(0, None), prominence=(0, None)):
        """
        window: the main peak intensity is the average of intensity of within window size
            centered around peak position.
        """
        if left is None:
            left = self.grid[0]
        if right is None:
            right = self.grid[-1]
        try: 
            peaks = self._peaks(
                height=1, left=left, right=right, 
                width=width, prominence=prominence
            )
            sorted_h = np.sort(peaks[:,-1])
            assert len(sorted_h) > 0
            if len(sorted_h) == 1:
                position, _, _, = peaks[0]
            elif (sorted_h[-1]-sorted_h[-2] < 0.2):
                # Pick the first peak if it is not higher than any other peak by at least 0.2
                position, _, _,  = peaks[0] # the first peak that satisfy all the filters.   
            else:
                # Otherwise pick the highest peak
                position, _, _, = peaks[np.argmax(peaks[:,-1])]                 

        except Exception:
            peaks = self._peaks(gradient=2, reverse=True, left=left, right=right)
            position, _, _, = peaks[np.argmin(peaks[:,-1])] # Used as an initial guess to calculate curvature
        self.main_peak["position"] = position
        select = (self.grid >= position - window/2) & (self.grid < position + window/2)
        self.main_peak["intensity"] = self.spec[select].mean()
        _, _, curvature, (grid,fit) = self._curve(guess=position, extremum="max", fit_range=None, window=4)
        self.main_peak["curvature"] = curvature
        self._main_peak_curve = (grid, fit)

    def find_intensity_at_energy(self, energy, window=1):
        """ If `energy`<100, find the intensity at `energy` above edge position instead.
        """
        self._energy = round(energy, 1)
        if self._energy < 100:
            _energy = self._energy + self.edge["position"]
            self._energy_position = _energy
        else:
            _energy = self._energy
        select = (self.grid >= _energy - window/2) & (self.grid < _energy + window/2)
        self.other[f"intensity_{self._energy:.1f}"] = self.spec[select].mean()
        
        
    def find_main_pit(self, curvature_window=None):
        left = self.edge["position"] + 20
        pits = self._peaks(left=left, reverse=True)
        try:
            position, _, intensity = pits[np.argmin(pits[:,-1])]
        except ValueError:
            select = self.grid > left
            index_min = np.argmin(self.spec[select])
            position = self.grid[select][index_min]
            intensity = self.spec[select][index_min]
        position, intensity, curvature, _ = \
            self._curve(guess=position, extremum="min", window=16)
        if curvature_window is not None:
            select = (self.grid > position - curvature_window/2) & (self.grid < position + curvature_window/2)
            curvature = np.abs(self._derivative(n=2)[select].mean())
        self.pit["position"] = position
        self.pit["intensity"] = intensity
        self.pit["curvature"] = curvature
    
    def find_fluctuation(self):
        left = self.main_peak["position"]
        select = (self.grid > left)
        self.other["fluctuation"] = np.abs(self._derivative(n=2)[select].mean())

        
    def find_last_peak(self):
        left = self.pit["position"]
        peaks = self._peaks(left=left, prominence=0.01)
        try: 
            position, _, intensity = peaks[0]
        except IndexError:
            position, intensity = self.grid[-1], self.spec[-1]
        position, intensity, curvature, (grid,fit) = \
        self._curve(guess=position, extremum="max", window=6)
        self.last_peak["position"] = position
        self.last_peak["intensity"] = intensity
        self.last_peak["curvature"] = curvature

    
    def find_pit_last_spread(self):
        self.other["pit_last_spread"] =  self.last_peak["intensity"] - self.pit["intensity"]
    
    def find_peak_separation(self):
        self.other["main_last_separation"] = self.last_peak["position"] - self.main_peak["position"]
        self.other["main_pit_separation"] = self.pit["position"] - self.main_peak["position"]
    
    def find_pre_peak(self):
        left = self.grid[0]+3
        right = self.edge["position"]
        select = (self.grid >= left) & (self.grid < right)
        try:
            peaks = self._peaks(left=left, right=right-1)
            position, _, intensity = peaks[np.argmax(peaks[:,-1])]
        except ValueError:
            try:
                peaks = self._peaks(left=left, right=right-3, reverse=True, gradient=2)
                position, _, intensity = peaks[np.argmax(peaks[:,1])]
            except ValueError:
                position, intensity = None, 0
        self.pre_peak["position"] = position
        self.pre_peak["intensity"] = intensity
       
#     def find_sec_peak(self):
#         left = self.main_peak["position"]
#         right = self.pit["position"] - 2
#         select = (self.grid >= left) & (self.grid < right)
#         peaks = self._peaks(left=left, right=right)
#         try:
#             position, _, intensity = peaks[-1]
#         except IndexError:
#             peaks = self._peaks(left=left, right=right, reverse=True, gradient=2, prominence=0.003)
#             try:
#                 position, curvature, intensity = peaks[-1]
#             except IndexError:
#                 position = (self.main_peak["position"] + self.pit["position"]) / 2
#                 pos_index = np.argmin(abs(self.grid - position))
#                 intensity = self.spec[pos_index]
            
#         self.sec_peak["position"] = position
#         self.sec_peak["intensity"] = intensity

    def find_sec_peak(self):
        left = self.main_peak["position"] + 5
        right = self.pit["position"] - 2
        select = (self.grid >= left) & (self.grid < right)
        peaks_2nd = self._peaks(left=left, right=right, reverse=True, gradient=2, prominence=0.003)
        try:
            position, curvature, intensity = peaks_2nd[np.argmax(peaks_2nd[:,-1])]
        except ValueError:
            position = (self.main_peak["position"] + self.pit["position"]) / 2
            pos_index = np.argmin(abs(self.grid - position))
            intensity = self.spec[pos_index]
            curvature = 0
            
        self.sec_peak["position"] = position
        self.sec_peak["intensity"] = intensity
        self.sec_peak["curvature"] = curvature
            
        
    def find_descriptors(self, features="all", energy=None):
        if "edge" in features or features == "all":
            self.find_edge()
        if "main_peak" in features or features == "all":
            self.find_main_peak()
        if "pit" in features or features == "all":
            self.find_main_pit()
        # if "sec_peak" in features or features == "all":
        #     self.find_second_peak()
        if "last" in features or features == "all":
            self.find_peak_last_pit()
        if "peak_separation" in features or features == "all":
            self.find_peak_separation()
        if "pre_peak" in features or features == "all":
            self.find_pre_peak()
        if energy is not None:
            self.find_intensity_at_energy(energy)
    
    def as_dict(self):
        descriptor_dict = {}
        for name, descriptor in self.__dict__.items():
            if name not in self.descriptors:
                continue
            self.descriptors[name] = descriptor
            for feature, value in descriptor.items():
                if (name == "other") & (value is not None):
                    display_name = f"{feature}"
                elif ((name=="edge") & (feature == "intensity") or (value is None)):
                    continue
                else:
                    display_name = f"{name}_{feature}"
                descriptor_dict[display_name] = value        
        return descriptor_dict

    def plot(self, ax=None, vlines=[], hlines=[]):
        
        # plot spectrum and arctan fit
        ax.plot(self.grid, self.spec)
        ax.plot(self.grid, self.arctan, lw=0.5, color='g')
        # plot edge position
        ax.plot(self.edge["position"], self.edge["intensity"], color='r', marker='o')
        # plot main peak
        ax.plot(self.main_peak["position"], self.main_peak["intensity"], color='r', marker='o')
        # plot pre peak
        try:
            ax.plot(self.pre_peak["position"], self.pre_peak["intensity"], color='r', marker='o')
        except:
            pass
        # plot intensity at a given energy
        try:
            ax.plot(self._energy_position, self.other[f"intensity_{self._energy:.0f}"], color='r', marker='o')
        except:
            pass
        # plot pit
        ax.plot(self.pit["position"], self.pit["intensity"],color='r', marker='o')
        # plot second peak
        try:
            ax.plot(self.sec_peak["position"], self.sec_peak["intensity"], color='r', marker='o')
        except:
            pass
        # plot peak after pit
        ax.plot(self.last_peak["position"], self.last_peak["intensity"], color='r', marker='o')
        for l in vlines:
            ax.axvline(l, color='k', alpha=0.5)
        for l in hlines:
            ax.axhline(l, oclor='k', alpha=0.5)
    
    def _curve(self, guess=None, extremum=None, fit_range=None, window=4):

        if guess is not None: # only fit around selected position
            select = (self.grid >= guess-window/2) & (self.grid < guess+window/2)
        elif fit_range is not None: # only fit within range
            select = (self.grid>=fit_range[0]) & (self.grid<fit_range[1])
        else: # use the whole range
            select = np.ones_like(self.grid, dytpe=bool)
        grid = self.grid[select]
        spec = self.spec[select]
        
        # fit the selected range
        polinom = Polynomial.fit(grid, spec, 2)
        fit = polinom(grid)

        # find the the peak/pit positions
        if extremum == "max":
            extreme_index = np.argmax(fit)
        elif extremum == "min":
            extreme_index = np.argmin(fit)
        else:
            a = polinom.convert().coef[2] # a in ax^2+bx+c
            extreme_index = np.argmax(fit) if a>0 else np.argmin(fit)

        position = grid[extreme_index]
        intensity = spec[extreme_index]

        spec_d2 = np.gradient(np.gradient(spec))
        curvature = abs(spec_d2[extreme_index])
        
        return position, intensity, curvature, (grid,fit)
            
    
    def _peaks(
        self, gradient=0, reverse=False,
        left=None, right=None,
        width=(0, None), height=0, prominence=0
    ):
        width = list(width)
        for i in [0,1]: # eV -> data points
            width[i] = None if width[i] is None else width[i] / (self.grid[1] - self.grid[0]) 

        if gradient:
            spec = self._derivative(n=gradient)
        else:
            spec = self.spec
        if reverse: spec = -spec
        peak_indices, properties = find_peaks(spec, height=height, prominence=prominence, width=width)
        peak_positions = self.grid[peak_indices]
        
        if left is None:
            left = self.grid[0]
        if right is None:
            right = self.grid[-1]
        select1 = (peak_positions >= left) & (peak_positions <= right)        
        select2 = True if width[1] is None else (properties["widths"] < width[1]) 
        drop1 = (peak_positions > right)

        select = select1 & select2 & (~drop1)

        return np.stack(
            [
                self.grid[peak_indices][select],
                spec[peak_indices][select], # derivatives in case gradient > 0
                self.spec[peak_indices][select]
            ], axis=1
        )
    
    def _derivative(self, n=1):
        derivative = self.spline.derivative(n=n)
        return derivative(self.grid)