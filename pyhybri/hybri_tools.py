import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from enum import Enum
import hashlib
import scipy.signal as scg
from numba import njit

# --- CONSTANTS ---
HEAD_RADIUS = 0.0875 # metri
NFREQS = 256 # n freqs for multiband filter
ETA = HEAD_RADIUS + 0.001 # minimum distance threshold
MAX_DELAY_SEC = 1 # max delay per sample

SLEW_RATE = 0.01 # smooth fractional delay

P_REF = 101325.0
T0 = 293.15

INTERNAL_KERNEL_TRANSITION = 0.003 # in sec.
MAX_DISTANCE_TRANSITION = 0.5

LRU_CAPACITY = 4096

HOT_GAMMA = 1.40
R_AIR = 287.05 # J/(KgK)
R_V = 461.5 # J/(KgK)

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

class CartesianPoint():
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.z = z
        self.y = y

    def to_polar(self):
        rho = np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        rho = max(rho, ETA)
        phi = np.arcsin(self.z / rho)
        theta = np.arctan2(self.y, self.x)
        theta = theta if theta >= 0 else theta + 2 * np.pi
        return PolarPoint(rho=rho, phi=phi, theta=theta, opt=AngleMode.RADIANS)

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

    def _get_hash(self, temp: float, hum: float, pres: float):
        return hashlib.md5(f"{self.rho}_{self.phi}_{self.theta}_{temp}_{hum}_{pres}".encode()).hexdigest()

class RData():
    def __init__(self, rir1: int, rir2: int, smooth_factor: float, source1: NDArray[np.complex64], source2: NDArray[np.complex64], morphed: NDArray[np.complex64]) -> None:
        self.rir1 = rir1
        self.rir2 = rir2
        self.smooth_factor = smooth_factor
        self.source1 = source1
        self.source2 = source2
        self.morphed = morphed

class AirData():
    def __init__(self, temperature: float, humidity: float, pressure: float = 101325.0):
        """
        AirData for ISO 9613-1

        Parameters
        ----------
        temperature : float
            tempertature in celsius
        humidity : float
            humidity in percent from 0 to 100 %
        pressure : float
            atmospheric pressure in Pa, default = 101325
        """
        
        self.celsius = temperature
        self.pres_pa = pressure

        self.kelvin = temperature + 273.15
        self.humidity = humidity
        self.pressure = pressure / P_REF
        

class HBuilded():
    def __init__(self, hrir: NDArray[np.float64], itd: float, gain: NDArray[np.float64]):
        self.hrir = hrir
        self.itd = itd
        self.gain = gain

@njit()
def multiply_spectrum(x: NDArray[np.complex64], y: NDArray[np.complex64]) -> NDArray[np.complex64]:
    return x * y

class ISO9613Filter():
    def __init__(self, air_data: AirData, fs: float):
        self.air_data = air_data
        self.fs = fs
        self.frequencies = np.linspace(0, self.fs / 2, NFREQS)
        self.fnorm = self.frequencies / (self.fs / 2) # normalized

    def get_attenuation_air_absorption(self) -> NDArray[np.float64]:
        p_sat = P_REF * (10 ** (-6.8346 * (273.16 / self.air_data.kelvin) ** 1.261 + 4.6151))
        h = self.air_data.humidity * (p_sat / (self.air_data.pressure * P_REF))
        tr = self.air_data.kelvin / T0
        tr_pos = tr ** 0.5
        tr_neg = tr ** -0.5

        # Frequenze di rilassamento per ossigeno e azoto (Hz)
        f_rO = self.air_data.pressure * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
        f_rN = self.air_data.pressure * tr_neg * (9.0 + 280.0 * h * np.exp(-4.17 * (tr ** (-1 / 3.0) - 1.0)))

        # Calcolo del coefficiente di assorbimento in dB/m
        freq_squared = self.frequencies ** 2

        # Termine per l'assorbimento classico (viscosità e conduzione termica)
        alpha_classical = (
            1.84e-11
            * (1 / self.air_data.pressure)
            * tr_pos
        )

        # Termine per il rilassamento dell'ossigeno
        alpha_oxygen = (
            0.01275
            * np.exp(-2239.1 / self.air_data.kelvin)
            * (f_rO + (freq_squared / f_rO)) ** -1
        )

        # Termine per il rilassamento dell'azoto
        alpha_nitrogen = (
            0.1068
            * np.exp(-3352.0 / self.air_data.kelvin)
            * (f_rN + (f_rN ** 2 + freq_squared)) ** -1
        )

        # Coefficiente di assorbimento totale (in dB/m)
        alpha_term = alpha_classical + tr ** (-2.5) * (alpha_oxygen + alpha_nitrogen)
        alpha = 8.686 * freq_squared * alpha_term
        return alpha.astype(np.float64)

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
        relative_distance = max(0.0, distance)
        db_attenuation = alpha_absortion * relative_distance
        gain = 10 ** (-db_attenuation / 20)
        gain[-1] = 0.0
        filtered = self.multiband_fft_filter(frame=frame, attenuation=gain)
        return filtered

class GeometricAttenuation():
    def __init__(self, fs: float, gamma: float, channels: int = 1) -> None:
        self.fs = fs
        self.gamma = gamma
        self.curr_delay = np.zeros(channels, dtype=float)
        self.max_delay_sample = int(MAX_DELAY_SEC * self.fs)

    def apply_fractional_delay(self, signal: NDArray[np.float64], distance: float, channel: int, sound_speed: float) -> float:
        channel_index = channel
        delay = distance * self.fs / sound_speed
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
        factor = (original_distance / distance) ** self.gamma # il fattore di attenuazione è in rapporto con la distanza originale in modo tale da lasciare invariata l'ampiezza alla distanza originale
        return factor

def woodworth_itd3d(point: PolarPoint, sound_speed: float) -> float:
    # theta = (point.theta + np.pi) % (2 * np.pi) - np.pi
    sin_theta = np.sin(-point.theta)
    cos_phi = np.cos(point.phi)
    svalue = point.rho * point.rho + HEAD_RADIUS * HEAD_RADIUS - 2 * HEAD_RADIUS * point.rho * sin_theta * cos_phi
    num = point.rho + HEAD_RADIUS * sin_theta * cos_phi - np.sqrt(svalue)
    return num / sound_speed

class RBuilded():
    def __init__(self, rir: NDArray[np.float64], power_spectrum: NDArray[np.float64], freqs: NDArray[np.float64], integr: NDArray[np.float64]):
        self.rir = rir
        self.power_spectrum = power_spectrum
        self.freqs = freqs
        self.integr = integr

def partitioned_convolution(x: NDArray[np.float64], prev_kernel: NDArray[np.float64]|None, curr_kernel: NDArray[np.float64], transition_length: int) -> NDArray[np.float64]:
    if prev_kernel is None:
        return scg.fftconvolve(x, curr_kernel, mode="full")

    ksize = max(prev_kernel.size, curr_kernel.size)
    total_size = x.size + ksize - 1
    prev_kernel = np.pad(prev_kernel, (0, ksize - prev_kernel.size), mode="constant")
    kernel_padded = np.pad(curr_kernel, (0, ksize - curr_kernel.size), mode="constant")

    transition_length = min(transition_length, total_size)
    smoothed = intermediate_segment(x=x, k1=prev_kernel, k2=kernel_padded, ksize=ksize, tlength=transition_length)
    if transition_length < x.size:
        rest_part = scg.fftconvolve(x[transition_length:], kernel_padded, mode="full")
        smoothed[transition_length:transition_length + rest_part.size] += rest_part
    return smoothed

@njit(cache=True)
def intermediate_segment(x: NDArray[np.float64], k1: NDArray[np.float64], k2: NDArray[np.float64], ksize: int, tlength: int) -> NDArray[np.float64]:
    segment = np.zeros(x.size + ksize - 1, dtype=np.float64)
    for i in range(tlength):
        alpha = float(i) / (tlength - 1.0)
        crossed = (1.0 - alpha) * k1 + alpha * k2
        current_x_sample = x[i]
        for k_idx in range(ksize):
            segment[i + k_idx] += current_x_sample * crossed[k_idx]
    return segment

@njit(cache=True)
def cross_fade(k1: NDArray[np.float64], k2: NDArray[np.float64], tlength: int) -> NDArray[np.float64]:
    kcross = k2.copy()
    for i in range(tlength):
        alpha_linear = float(i) / (tlength - 1.0)
        alpha = alpha_linear * (np.pi / 2.0)
        a = np.cos(alpha)
        b = np.sin(alpha)
        if k1.ndim > 1:
            kcross[i, 0] = a * k1[i, 0] + b * k2[i, 0]
            kcross[i, 1] = a * k1[i, 1] + b * k2[i, 1]
        else:
            kcross[i] = a * k1[i] + b * k2[i]
    if tlength < k1.size:
        if k1.ndim > 1:
            kcross[tlength:, :] = k2[tlength:, :]
        else:
            kcross[tlength:] = k2[tlength:]
    return kcross

class Node[T]():
    def __init__(self, key: str, value: T) -> None:
        self.key = key
        self.value = value
        self.prev_node = None
        self.next_node = None

class LRUCache[T]():
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache = {}
        self.head = Node(key="HEAD", value=None)
        self.tail = Node(key="TAIL", value=None)
        self.head.next_node = self.tail
        self.tail.prev_node = self.head

    def __remove(self, node: Node) -> None:
        prev = node.prev_node
        nxt = node.next_node
        prev.next_node = nxt
        nxt.prev_node = prev

    def __add(self, node: Node) -> None:
        node.next_node = self.head.next_node
        node.prev_node = self.head
        self.head.next_node.prev_node = node
        self.head.next_node = node

    def __move_to_head(self, node: Node) -> None:
        self.__remove(node=node)
        self.__add(node=node)

    def put(self, key: str, value: T) -> None:
        if key in self.cache:
            self.__move_to_head(node=self.cache[key])
        else:
            node = Node(key=key, value=value)
            self.cache[key] = node
            self.__add(node=node)

        if len(self.cache) > self.capacity:
            lru = self.tail.prev_node
            self.__remove(node=lru)
            del self.cache[lru.key]

    def get(self, key: str) -> T|None:
        if key in self.cache:
            node = self.cache[key]
            self.__move_to_head(node=node)
            return node.value
        else:
            return None

def get_sound_speed(temp_celsius: float, rh: float, pres_pa: float) -> float: # sqrt(gamma * P / p)
    kelv = temp_celsius + 273.15
    p_sat = 6.1078 * pow(10, 7.5 * temp_celsius / (temp_celsius + 237.3)) * 100.0
    p_v = rh * p_sat
    rho = (pres_pa - p_v) / (R_AIR * kelv) + p_v / (R_V * kelv)
    c = np.sqrt(HOT_GAMMA * pres_pa / rho)    
    return c
    
# ===================================
# ===================================
# ===================================

class LinearTrajectory():
    def __init__(self, cpa: NDArray[np.float32], direction: NDArray[np.float32]) -> None:
        """
        SET TRAJECTORY

        Parameters
        ----------
        cpa : NDArray[float]
            at index [0] x = front-back meters
            at index [1] y = left-right meters
            at index [2] z = up-down meters

        direction : NDArray[float]
        """

        self.cpa = cpa
        norm = np.linalg.norm(direction)
        if norm == 0:
            print("[ERROR] Direction vector must be not zero")
            exit(1)

        self.direction = direction / norm

    def get_points(self, distances: NDArray[np.float32]) -> tuple[list[PolarPoint], list[CartesianPoint]]:
        point = self.cpa + self.direction * distances[:, None] # Formula: P0 + V * t
        points_p = []
        points_c = []
        for p in point:
            c = CartesianPoint(p[0], p[1], p[2])
            points_p.append(c.to_polar())
            points_c.append(c)
        return (points_p, points_c)

class ParametricTrajectory():
    @staticmethod
    def get_circular_points(
        omega: NDArray[np.float32], 
        t: NDArray[np.float32], 
        radius: NDArray[np.float32], 
        start_elevation: float, 
        vertical_vel: NDArray[np.float32]
    ) -> tuple[list[PolarPoint], list[CartesianPoint]]:
        
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        z = start_elevation + vertical_vel * t
        stack = np.stack((x, y, z), axis=-1)
        points_p = []
        points_c = []
        for p in stack:
            c = CartesianPoint(p[0], p[1], p[2])
            points_p.append(c.to_polar())
            points_c.append(c)
        return (points_p, points_c)

def linear_direction_info(direction: NDArray[np.float32]) -> None:
    rho = np.linalg.norm(direction)
    if rho == 0:
        print("[ERROR] Zero norm!")
        exit(1)
    azi = np.rad2deg(np.arctan2(direction[1], direction[0]))
    ele = np.rad2deg(np.arcsin(direction[2] / rho))
    print("[DIRECTION INFO]")
    print(f"[AZIMUTH VARIATION] = {azi:.3f}\u00B0")
    print(f"[ELEVATION VARIATION] = {ele:.3f}\u00B0")
