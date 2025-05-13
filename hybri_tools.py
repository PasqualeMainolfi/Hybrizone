import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from enum import Enum
import hashlib
from numba import njit

SOUND_SPEED = 343.3 # sound speed
NFREQS = 256 # n freqs for multiband filter
GAMMA = 0.7 # exponent in distance perception
ETA = 0.1 # minimum distance threshold
MAX_DELAY_SEC = 1 # max delay per sample
SLEW_RATE = 0.01 # smooth fractional delay

class CurveMode(Enum):
    LINEAR = 0
    SIGMOID = 1
    LOGARITHMIC = 2
    EXPONENTIAL = 3

class CoordMode(Enum):
    INTERAURAL = 0
    REGULAR = 1

class AngleMode(Enum):
    RADIANS = 0
    DEGREE = 1

class BuildMode(Enum):
    INVERSE_DISTANCE = 1
    LINEAR_INVERSE_DISTANCE = 2
    LINEAR = 3
    
class InterpolationDomain(Enum):
    TIME = 0
    FREQUENCY = 1

class CartesianPoint():
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.z = z
        self.y = y

class PolarPoint():
    def __init__(self, rho: float, phi: float, theta: float, opt: AngleMode):
        
        """
        Polar Point

        Parameters
        ----------
        rho : float
            radius in meters (0, ...)
        phi : float
            elevation (-90 <= phi <= 90) in degree
        theta : float
            azimuth (0 <= theta <= 360) in degree
        """

        self.opt = opt
        self.rho = rho
        
        if opt == AngleMode.DEGREE:
            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)
        
        self.phi = phi
        self.theta = theta
        
    def get_cartesian(self, mode: CoordMode) -> CartesianPoint:
        x = 0.0
        y = 0.0
        z = 0.0
        
        match mode:
            case CoordMode.REGULAR:
                x = np.sin(self.theta) * np.cos(self.phi)
                y = np.cos(self.theta) * np.cos(self.phi)
                z = np.sin(self.phi)
            case CoordMode.INTERAURAL:
                x = np.sin(self.theta)
                y = np.cos(self.theta) * np.cos(self.phi)
                z = np.cos(self.theta) * np.sin(self.phi)
            case _:
                print("[ERROR] Mode not allowed!")
                exit(1)
        
        x = x * 0.99
        y = y * 0.99 + 0.01
        z = z * 0.99
        
        return CartesianPoint(x=x, y=y, z=z)
    
    def _get_hash(self):
        return hashlib.md5(f"{self.rho}_{self.phi}_{self.theta}".encode()).hexdigest()

class AirData():
    def __init__(self, temperature: float, humidity: float, pressure: float = 101325.0):
        """
        AirData for ISO 9613-1

        Parameters
        ----------
        temperature : float
            tempertature in celsius
        humidity : float
            humidity in percent from 0 to 1 where 1 = 100%
        pressure : float
            atmospheric pressure in Pa, default = 101325
        """
        
        self.kelvin = temperature + 273.15
        self.humidity = humidity
        self.pressure = pressure / 101325.0

class HBuilded():
    def __init__(self, hrir: NDArray[np.float64], itd: float):
        self.hrir = hrir
        self.itd = itd

@njit()
def multiply_spectrum(x: NDArray[np.complex64], y: NDArray[np.complex64]) -> NDArray[np.complex64]:
    return x * y

class ISO9613Filter():
    def __init__(self, air_data: AirData, fs: float):
        self.air_data = air_data
        self.fs = fs
        self.frequencies = np.linspace(0, fs / 2, NFREQS)
        self.fnorm = self.frequencies / (self.fs / 2) # normalized
    
    def get_attenuation_air_absorption(self) -> NDArray[np.float32]:
        h = self.air_data.humidity * 10 ** ((-6.8346 * (273.16 / self.air_data.kelvin) ** 1.261) + 4.6151)
        
        # Frequenze di rilassamento per ossigeno e azoto (Hz)
        f_rO = self.air_data.pressure * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h)) / 1000.0
        f_rN = self.air_data.pressure * (9.0 + 280.0 * h * np.exp(-4.17 * ((self.air_data.kelvin / 293.0) - 1))) / 1000.0
        
        # Calcolo del coefficiente di assorbimento in dB/m
        freq_kHz = self.frequencies / 1000.0  # conversione in kHz
        freq_kHz = freq_kHz ** 2
        
        # Termine per l'assorbimento classico (viscosità e conduzione termica)
        alpha_classical = 1.84e-11 * (self.air_data.pressure) ** (-1) * np.sqrt(self.air_data.kelvin / 293.15) * freq_kHz
        
        # Termine per il rilassamento dell'ossigeno
        alpha_oxygen = 0.01275 * np.exp(-2239.1 / self.air_data.kelvin) * freq_kHz / (f_rO + (freq_kHz / f_rO))
        
        # Termine per il rilassamento dell'azoto
        alpha_nitrogen = 0.1068 * np.exp(-3352.0 / self.air_data.kelvin) * freq_kHz / (f_rN + (freq_kHz / f_rN))
        
        # Coefficiente di assorbimento totale (in dB/m)
        alpha = alpha_classical + alpha_oxygen + alpha_nitrogen
        
        return alpha.astype(np.float32)
    
    def multiband_fft_filter(self, frame: NDArray[np.float32], attenuation: NDArray[np.float32]):
        # frame is hrir
        n = len(frame)
        gain_interpolator = interp1d(self.fnorm, attenuation, bounds_error=False, fill_value=(attenuation[0], attenuation[-1]))
        fft_freqs = np.linspace(0, 1, n // 2 + 1)
        fresp = gain_interpolator(fft_freqs)
        fresp = fresp.astype(np.complex64)
        
        filtered = None
        if frame.ndim > 1:
            fresp = fresp[:, None]
            fft = np.fft.rfft(frame, axis=0)
            filtered_fft = multiply_spectrum(fft, fresp)
            filtered = np.fft.irfft(filtered_fft, n=n, axis=0)
        else:
            fft = np.fft.rfft(frame)
            filtered_fft = filtered_fft = multiply_spectrum(fft, fresp)
            filtered = np.fft.irfft(filtered_fft, n=n)
        
        return filtered

    def air_absorption_filter(self, frame: NDArray[np.float64], alpha_absortion: NDArray[np.float64], distance: float) -> NDArray[np.float64]:
        relative_distance = distance if distance > 0 else 0
        db_attenuation = alpha_absortion * relative_distance
        gain = np.exp2(-db_attenuation / (20 * np.log10(2)))
        gain[-1] = 0.0
        filtered = self.multiband_fft_filter(frame=frame, attenuation=gain)
        return filtered

class GeometricAttenuation():
    def __init__(self, fs: float, channels: int = 1) -> None:
        self.fs = fs
        self.curr_delay = np.zeros(channels, dtype=float)
        self.max_delay_sample = int(MAX_DELAY_SEC * self.fs)
        
    def apply_fractional_delay(self, signal: NDArray[np.float64], distance: float, channel: int) -> float:
        channel_index = channel
        delay = distance * self.fs / SOUND_SPEED
        delta_delay = delay - self.curr_delay[channel_index]
        delta_delay = np.clip(delta_delay, -SLEW_RATE, SLEW_RATE) # Controlla la pendenza (slew rate) della variazione di delay in campioni per campione
        self.curr_delay[channel_index] += delta_delay
        
        n = len(signal)
        indexes = np.arange(n)
        delayed_indexes = indexes - self.curr_delay[channel]
        interpolator = interp1d(indexes, signal, kind="linear", bounds_error=False, fill_value=0.0)
        return interpolator(delayed_indexes)
        
    def calculate_geometric_attenuation(self, source_distance: float, distance: float) -> float:
        original_distance = max(source_distance, ETA)
        factor = (original_distance / distance) ** GAMMA # il fattore di attenuazione è in rapporto con la distanza originale in modo tale da lasciare invariata l'ampiezza alla distanza originale
        return factor