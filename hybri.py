import scipy.signal
import scipy.special
import hrir_builder as hrb
from hybri_tools import CoordMode, PolarPoint, BuildMode, InterpolationDomain, AirData, CurveMode, HBuilded, RBuilded 
from hybri_tools import AngleMode # noqa
from hrir_builder import HInfo
import rir_builder as rib
import numpy as np
from numpy.typing import NDArray
import scipy
from concurrent.futures import ThreadPoolExecutor
import time
from numba import njit

TRANSITION_FACTOR = 0.5
MAX_TRANSITION_SAMPLES = 512
SOFT_CLIP_SCALE = 1.0 / 0.707

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

class SmoothedConvolution():
    
    @staticmethod
    def apply_intermediate(x: NDArray[np.float64], prev_kernel: NDArray[np.float64]|None, curr_kernel: NDArray[np.float64], transition_length: int) -> NDArray[np.float64]:
        if prev_kernel is None:
            return scipy.signal.fftconvolve(x, curr_kernel, mode="full")
        
        ksize = max(prev_kernel.size, curr_kernel.size)
        total_size = x.size + ksize - 1
        prev_kernel = np.pad(prev_kernel, (0, ksize - prev_kernel.size), mode="constant")
        kernel_padded = np.pad(curr_kernel, (0, ksize - curr_kernel.size), mode="constant")
        
        transition_length = min(transition_length, total_size)
        smoothed = intermediate_segment(x=x, k1=prev_kernel, k2=kernel_padded, ksize=ksize, tlength=transition_length)
        if transition_length < x.size:
            rest_part = scipy.signal.fftconvolve(x[transition_length:], kernel_padded, mode="full")
            smoothed[transition_length:transition_length + rest_part.size] += rest_part
        return smoothed

class HybriParams():
    def __init__(
        self, 
        hrir_database_path: str, 
        rir_database_path: str | None, 
        coord_mode: CoordMode, 
        interp_domain: InterpolationDomain, 
        build_mode: BuildMode, 
        chunk_size: int = 1024, 
        interpolation_neighs: int = 3, 
        sample_rate: float = 44100
    ) -> None:
        
        """
        HYBRIZONE INIT PARAMETERS

        Parameters
        ----------
        hrir_database_path_path : str
            path to HRIRs .h5 database
        rir_database_path : str
            path to RIRs .h5 database. If null IR will not applied
        coord_mode : CoordMode
            coordinates mode (see CoordMode)
        interp_domain : InterpDomain
            HRIRs interpolation domain (see InterpolationDomain)
        build_mode : BuildMode
            HRIRs interpolation method
        chunk_size : int
            chunk size in samples for real-time buffering overlap and save. by default, 1024
        interpolation_neighs : int
            how many closest HRIRs in interpolation process. by default, 3
        sample_rate : float
            sample rate in Hz. by default 44100
        """

        self.hrir_database_path = hrir_database_path
        self.rir_database_path = rir_database_path
        self.coord_mode = coord_mode
        self.interpolation_domain = interp_domain
        self.build_mode = build_mode
        self.chunk_size = chunk_size
        self.interpolation_neighs = interpolation_neighs
        self.fs = sample_rate

class RTOverlapSaveBufferConvolution():
    def __init__(self, chunk: int):
        self.buffer = np.empty(0, dtype=np.float64)
        self.chunk = chunk
        self.pkernel = None
        self.pbuffer = None
        self.transition_size = int(chunk * TRANSITION_FACTOR) if chunk < 2048 else MAX_TRANSITION_SAMPLES

    def process(self, x: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
        convolution = SmoothedConvolution.apply_intermediate(x=x, prev_kernel=self.pkernel, curr_kernel=kernel, transition_length=self.transition_size)
        self.pkernel = kernel.copy()

        max_lenght = min(self.buffer.size, convolution.size)
        convolution[:max_lenght] += self.buffer[:max_lenght]
        convolved = convolution[:self.chunk].astype(np.float64)
        self.buffer = convolution[self.chunk:]
        return convolved

class HybriKernels():
    def __init__(self, rir: NDArray[np.float64] | None, hrir: NDArray[np.float64], itd: float, gain: NDArray[np.float64]) -> None:
        self.rir = rir
        self.hrir = hrir
        self.itd = itd
        self.gain = gain

class Hybrizone():

    def __init__(self, params: HybriParams) -> None:
        self.__params = params
        self.hrir_builder = hrb.HRIRBuilder(hrir_database=params.hrir_database_path, mode=params.coord_mode, interp_domain=params.interpolation_domain)

        self.rir_builder = None
        self.__rir_buffer = None
        if self.__params.rir_database_path is not None:
            self.rir_builder = rib.RIRMorpha(
                rir_database_path=str(params.rir_database_path), 
                source_distance=self.hrir_builder.dataset.get_source_distance()
            )
            self.__rir_buffer = RTOverlapSaveBufferConvolution(chunk=self.__params.chunk_size)

        self.__hrir_left_buffer = RTOverlapSaveBufferConvolution(chunk=self.__params.chunk_size)
        self.__hrir_right_buffer = RTOverlapSaveBufferConvolution(chunk=self.__params.chunk_size)
        
        self.__temp_hrir = None
        self.__temp_rho = None
        self.__temp_rir = None
        
        self.__cache_hrir = {}
        
        self.__htime = 0.0
        self.__rtime = 0.0
        self.__ptime = 0.0
        self.__counter = 1

    def close(self) -> None:
        """
        CLOSE HYBRIZONE
        """
        print("[INFO] Close Hybrizone...")
        print("[INFO] Free memory")
        self.hrir_builder.close()
        if self.rir_builder is not None:
            self.rir_builder.close()
        self.__temp_hrir = None
        self.__temp_rho = None
        self.__temp_rir = None
        
        self.__htime = 0.0
        self.__rtime = 0.0
        self.__ptime = 0.0
        self.__counter = 1
        
        print("[INFO] Hybrizone closed!")

    def process_frame(self, frame: NDArray[np.float64], kernels: HybriKernels) -> NDArray[np.float32]:
        """
        APPLY RIR AND HRIR

        Parameters
        ----------
        frame : NDArray[np.float64]
            frame to be process
        kernels : HybriKernels
            RIR and HRIR kernel

        Returns
        -------
        NDArray[np.float64]
            frame processed
        """

        tstart = time.perf_counter()
        mono = frame
        if kernels.rir is not None:
            mono = self.__rir_buffer.process(x=frame, kernel=kernels.rir)

        with ThreadPoolExecutor(max_workers=2) as convolver:
            lc = convolver.submit(self.__hrir_left_buffer.process, mono, kernels.hrir[:, 0])
            lr = convolver.submit(self.__hrir_right_buffer.process, mono, kernels.hrir[:, 1])
            left_hrir = lc.result()
            right_hrir = lr.result()

        convolved = np.column_stack((left_hrir, right_hrir)).astype(np.float32)
        tend = time.perf_counter()
        self.__ptime += tend - tstart
        self.__counter += 1
        return np.tanh(convolved * SOFT_CLIP_SCALE)

    def set_position(self, position: PolarPoint) -> None:
        pkey = position._get_hash()

        if pkey in self.__cache_hrir:
            self.__temp_hrir = self.__cache_hrir[pkey]
        else:
            tstart = time.perf_counter()
            hrirs = self.query_hrirs(spatial_position=position, n_neighs=self.__params.interpolation_neighs)
            temp_hrir = self.build_distance_based_hrir(hrirs=hrirs)
            tend = time.perf_counter()
            self.__htime += tend - tstart            
            self.__cache_hrir[pkey] = temp_hrir
            self.__temp_hrir = temp_hrir

        self.__temp_rho = position.rho
    
    def set_morph_data(self, direction: float, morph_curve: CurveMode) -> None:
        tstart = time.perf_counter()
        self.__temp_rir = self.build_hybrid_space(direction=direction, morph_curve=morph_curve, rho=self.__temp_rho)
        tend = time.perf_counter()
        self.__rtime += tend - tstart
        
    def get_kernels(self) -> HybriKernels:
        return HybriKernels(rir=self.__temp_rir, hrir=self.__temp_hrir.hrir, itd=self.__temp_hrir.itd, gain=self.__temp_hrir.gain)
    
    def get_proc_times(self) -> dict:
        return {
            "htime": self.__htime / self.__counter,
            "rtime": self.__rtime / self.__counter,
            "ptime": self.__ptime / self.__counter
        }
    
    # --- START HRIR SECTION ---

    def imposed_air_conditions(self, air_data: AirData) -> None:
        """
        SET AIR CONDITIONS

        Parameters
        ----------
        air_data : AirData
            set air conditions: temperature, humidity and pressure (see AirData)
        """

        self.hrir_builder.set_air_conditions(air_data=air_data)
        if self.rir_builder is not None:
            self.rir_builder.set_air_conditions(air_data=air_data)


    def query_hrirs(self, spatial_position: PolarPoint, n_neighs: int) -> HInfo:
        """
        FIND NEAR HRIRs

        Parameters
        ----------
        spatial_position : PolarPoint
            spatial position (see PolarPoint)
        n_neighs : int
            number of neighs for HRIRs interpolation

        Returns
        -------
        HInfo
            info about selected HRIRs (see HInfo)
        """

        return self.hrir_builder.prepare_hrirs(point=spatial_position, neighs=n_neighs)

    def build_distance_based_hrir(self, hrirs: HInfo) -> HBuilded:
        """
        BUILD HRIR FROM HInfo (SELECTED HRIRs)

        Parameters
        ----------
        hrirs : HInfo
            info about selected HRIRs

        Returns
        -------
        HBuilded : 
            hrir, itd
        
        """

        return self.hrir_builder.build_hrir(hrirs_info=hrirs, method=self.__params.build_mode)

    def display_hrir(self, hrir: NDArray[np.float64], title: str) -> None:
        """
        PLOT HRIR ANALYSIS

        Parameters
        ----------
        hrir : NDArray[np.float64]
            hrir
        title : str
            plot title
        """

        self.hrir_builder.plot_hrir(data=hrir, title=title)

    # --- END HRIR SECTION ---

    # --- START RIR SECTION ---

    def set_rirs(self, rir1: int, rir2: int, smooth_factor: float = 0.1) -> None:
        """
        SET RIRs

        Parameters
        ----------
        rir1 : int
            index to RIR 1 (index of IR-keys in database attrs)
        rir2 : int
            index to RIR 2 (index of IR-keys in database attrs)
        smooth_factor : float, optional
            spectral envelope smooth factor, by default 0.1
        """

        tstart = time.perf_counter()
        self.rir_builder.set_rirs(rir1=rir1, rir2=rir2, smooth_factor=smooth_factor)
        tend = time.perf_counter()
        self.__rtime += tend - tstart
        self.__ptime += tend - tstart

    def set_rir1(self, rir1: int) -> None:
        """
        SET RIR 1

        Parameters
        ----------
        rir1 : int
            index to RIR 1 (index of IR-keys in database attrs)
        """
        self.rir_builder.set_rir1(rir1=rir1)

    def set_rir2(self, rir2: int) -> None:
        """
        SET RIR 2

        Parameters
        ----------
        rir2 : int
            index to RIR 2 (index of IR-keys in database attrs)
        """

        self.rir_builder.set_rir2(rir2=rir2)

    def get_rir_key(self, rir_index: int) -> str:
        """
        GET RIR KEY FROM DATABASE

        Parameters
        ----------
        rir_index : int
            rir index

        Returns
        -------
        str
            RIR name
        """

        return self.rir_builder.dataset.attrs["IR-keys"][rir_index]

    def set_smooth_factor(self, smooth_factor: float) -> None:
        """
        SET SMOOTH FACTOR

        Parameters
        ----------
        smooth_factor : float, optional
            spectral envelope smooth factor, by default 0.1
        """

        self.rir_builder.set_smooth_factor(smooth_factor=smooth_factor)

    def build_hybrid_space(self, direction: float, morph_curve: CurveMode, rho: float) -> NDArray[np.float64]:
        """
        BUILD HYBRID SPACE

        Parameters
        ----------
        direction : float
            morphing direction (0 to 1). 0 means only rir1, 0.5 morphed rir, 1.0 only rir 2.
        morph_curve : CurveMode
            transition curve (see CurveMode)

        Returns
        -------
        NDArray[np.float64]
            RIR
        """

        return self.rir_builder.morpha(direction=direction, morph_curve=morph_curve, distance=rho)

    def display_rir(self, rir: int|NDArray[np.float64], title: str|None = None) -> None:
        """
        _summary_

        Parameters
        ----------
        rir : int | NDArray[np.float64]
            RIR defined starting from database index or data array
        title : str | None, optional
            plot title. If rir=int plot title will be the RIR key, by default None
        """

        self.rir_builder.plot_rir(rir=rir, title=title)

    def get_rir(self, key: str) -> NDArray[np.float64]:
        return self.rir_builder.dataset[key][:]

    def get_rir_data(self, rir: NDArray[np.float64]) -> RBuilded:
        data = self.rir_builder._get_data_plot(rir=rir)
        return RBuilded(rir=rir, power_spectrum=data.mag, freqs=data.f, integr=data.integr)
    
    # --- END RIR SECTION ---
