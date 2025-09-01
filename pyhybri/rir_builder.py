import numpy as np
from numpy.typing import NDArray
import h5py
import hashlib

import scipy.fft
import scipy.signal
from hybri_tools import ISO9613Filter, GeometricAttenuation, AirData, ETA, CurveMode, RData, cross_fade, LRUCache
from hybri_tools import INTERNAL_KERNEL_TRANSITION, MAX_DISTANCE_TRANSITION, LRU_CAPACITY
from numba import njit
import scipy

class PlotData():
    def __init__(self, t: NDArray[np.float32], f: NDArray[np.float32], mag: NDArray[np.float32], integr: NDArray[np.float32]):
        self.t = t
        self.f = f
        self.mag = mag
        self.integr = integr

class MorpData():
    def __init__(self, direction: float, morph_curve: CurveMode):
        self.direction = direction
        self.morph_curve = morph_curve
        
    def _get_hash(self):
        return hashlib.md5(f"{self.direction}-{self.morph_curve}".encode()).hexdigest()

@njit()
def get_morphed_data(curve_value: float, source1: NDArray[np.complex64], source2: NDArray[np.float64], morphed: NDArray[np.float64]) -> NDArray[np.complex64]:
    sx = max(1.0 - 2.0 * curve_value, 0.0)  
    cx = 1.0 - abs(1.0 - 2.0 * curve_value)
    dx = max(2.0 * curve_value - 1.0, 0.0)
    return sx * source1 + cx * morphed + dx * source2

class RIRMorpha():
    def __init__(self, rir_database_path: str, source_distance: float, gamma: float):
        """
        Parameters
        ----------
        rir_database : str
            path to h5 database
        """
        
        self.dataset = h5py.File(rir_database_path, "r")
        self.fs = self.dataset.attrs["fs"]
        self.length = None
        
        self.iso9613 = None
        self.db_attenuation = None
        self.source_distance = source_distance
        
        self.geometric_attenuation = GeometricAttenuation(fs=self.fs, channels=1, gamma=gamma)
        
        self.__cache_rir_builded = LRUCache[RData](capacity=LRU_CAPACITY)
        self.__current_key = None

        self.source1 = None
        self.source2 = None
        self.morphed = None
        self.cache_data = { "r1": None, "r2": None, "sf": None }
        
        self.__prev_dist = None

    def close(self) -> None:
        self.dataset.close()       
    
    def set_air_conditions(self, air_data: AirData) -> None:
        self.iso9613 = ISO9613Filter(air_data=air_data, fs=self.fs)
        self.db_attenuation = self.iso9613.get_attenuation_air_absorption()
    
    def set_rirs(self, rir1: int, rir2: int, smooth_factor: float = 0.1) -> None:
        """
        SET RIRs

        Parameters
        ----------
        rir1 : int
            index to rir 1 (index of IR-keys in database attrs)
        rir2 : int
            index to rir 2 (index of IR-keys in database attrs)
        smooth_factor : float, optional
            spectral envelope smooth factor, by default 0.1
        """
        
        self.__current_key = hashlib.md5(f"{rir1}_{rir2}_{smooth_factor}_{self.iso9613.air_data.kelvin}_{self.iso9613.air_data.humidity}_{self.iso9613.air_data.pressure}".encode()).hexdigest()
        
        if self.__cache_rir_builded.get(key=self.__current_key) is None:
            k1 = self.dataset.attrs["IR-keys"][rir1]
            k2 = self.dataset.attrs["IR-keys"][rir2]
            source1 = self.dataset[k1][:]
            source2 = self.dataset[k2][:]
            n1 = len(source1)
            n2 = len(source2)
            
            if n1 < n2: 
                source1 = np.pad(source1, (0, n2 - n1), constant_values=0.0, mode="constant")
            elif n1 > n2: 
                source2 = np.pad(source2, (0, n1 - n2), constant_values=0.0, mode="constant")
            self.length = max(n1, n2)
            
            scep = self.__get_spectral_envelope(source=source1, smooth_factor=smooth_factor)
            tcep = self.__get_spectral_envelope(source=source2, smooth_factor=smooth_factor)
            source1f = np.fft.rfft(source1)
            source2f = np.fft.rfft(source2)
            target_flatten = source2f / (tcep + 1e-12)
            morphed = scep * target_flatten
            
            self.source1 = source1f
            self.source2 = source2f
            self.morphed = morphed
            self.cache_data["r1"] = rir1
            self.cache_data["r2"] = rir2
            self.cache_data["sf"] = smooth_factor
            rdata = RData(rir1=rir1, rir2=rir2, smooth_factor=smooth_factor, source1=source1f, source2=source2f, morphed=morphed)
            self.__cache_rir_builded.put(key=self.__current_key, value=rdata)
        else:
            rdata = self.__cache_rir_builded.get(key=self.__current_key)
            self.source1 = rdata.source1
            self.source2 = rdata.source2
            self.morphed = rdata.morphed
            self.cache_data["r1"] = rdata.rir1
            self.cache_data["r2"] = rdata.rir2
            self.cache_data["sf"] = rdata.smooth_factor
            
    def set_rir1(self, rir1: int) -> None:
        self.set_rirs(rir1=rir1, rir2=self.cache_data["r2"], smooth_factor=self.cache_data["sf"])
    
    def set_rir2(self, rir2: int) -> None:
        self.set_rirs(rir1=self.cache_data["r1"], rir2=rir2, smooth_factor=self.cache_data["sf"])
    
    def set_smooth_factor(self, smooth_factor: float) -> None:
        self.set_rirs(rir1=self.cache_data["r1"], rir2=self.cache_data["r2"], smooth_factor=smooth_factor)
        
    def __get_spectral_envelope(self, source: NDArray[np.float32], smooth_factor: float) -> NDArray[np.float32]:
        fft_source = np.fft.rfft(source)
        mag = np.abs(fft_source)
        log = np.log10(mag + 1e-12)
        realcp = np.fft.irfft(log).real
        realcp = np.fft.rfft(realcp).real
        realcp_mean = np.mean(realcp)
        realcp = np.exp(realcp - realcp_mean)
        
        kernel_length = int(len(realcp) * smooth_factor)
        kernel = np.ones(kernel_length) / kernel_length
        rc_smoothed = scipy.signal.fftconvolve(realcp, kernel, mode="same")
        
        scale_factor = np.max(mag) / (np.max(rc_smoothed) + 1e-12)
        return rc_smoothed * scale_factor
    
    def __nonlinear_morphing_curve(self, direction: float, curve_type: CurveMode) -> float:
        match curve_type:
            case CurveMode.LINEAR: 
                return direction
            case CurveMode.SIGMOID: 
                return 1 / (1 + np.exp(-10 * (direction - 0.5)))
            case CurveMode.EXPONENTIAL: 
                return direction**2
            case CurveMode.LOGARITHMIC: 
                return np.log10(direction * 9 + 1)
            case _:
                print("[ERROR] Curve mode not implemented!")
                exit(1)
    
    def __distance_based_rir(self, rir: NDArray[np.float64], rho: float) -> NDArray[np.float64]:
        factor = self.geometric_attenuation.calculate_geometric_attenuation(source_distance=self.source_distance, distance=rho)
        factor = factor if factor <= 1.0 else 1.0
        filtered = self.iso9613.air_absorption_filter(frame=rir, alpha_absortion=self.db_attenuation, distance=rho - max(self.source_distance, ETA)) * factor
        return filtered
    
    def morpha(self, direction: float, morph_curve: CurveMode, distance: float) -> NDArray[np.float32]:
        """
        RIRs MORPHING

        Parameters
        ----------
        direction : float
            morphing direction (0 to 1). 0 means only rir1, 0.5 morphed rir, 1.0 only rir 2.
        morph_curve : CurveMode
            transition curve (see CurveMode)

        Returns
        -------
        NDArray[np.float32]
            data rir
        """
        
        if self.__current_key is None:
            print("[ERROR] Set RIRs first!")
            exit(1)
        
        mf = self.__nonlinear_morphing_curve(direction=direction, curve_type=morph_curve)
        yspectrum = get_morphed_data(curve_value=mf, source1=self.source1, source2=self.source2, morphed=self.morphed)
        y = np.fft.irfft(yspectrum)
        y /= np.max(np.abs(y) + 1e-12)
        y_dist = self.__distance_based_rir(rir=y, rho=distance)
        y_out = y_dist
        
        if self.__prev_dist is not None:
            d = abs(distance - self.__prev_dist)
            if d > MAX_DISTANCE_TRANSITION:
                tlength = int(INTERNAL_KERNEL_TRANSITION * self.fs)
                y_out = cross_fade(k1=self.__prev_dist_rir, k2=y_dist, tlength=tlength)
            
        self.__prev_dist = distance
        self.__prev_dist_rir = y_dist
        return y_out
        
    def _get_data_plot(self, rir: NDArray[np.float32]) -> PlotData:
        n = len(rir)
        e = np.cumsum(rir[::-1] ** 2)[::-1]
        e /= np.max(e)
        db = 20 * np.log10(e + 1e-12)
        mag = np.abs(np.fft.rfft(rir))
        t = np.arange(n) / self.fs
        freqs = np.fft.rfftfreq(len(t), d=1 / self.fs)
        return PlotData(t=t, f=freqs, mag=mag, integr=db)
