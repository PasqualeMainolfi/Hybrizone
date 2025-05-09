import scipy.signal
import scipy.special
import hrir_builder as hrb
from hybri_tools import CoordMode, PolarPoint, BuildMode, InterpolationDomain, AirData, CurveMode
from hybri_tools import AngleMode # noqa
from hrir_builder import HInfo
from rir_builder import MorpData
import rir_builder as rib
import numpy as np
from numpy.typing import NDArray
import scipy
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

TRANSITION_FACTOR = 2 ** 4
SOFT_CLIP_SCALE = 1.0 / 0.707

class SmoothedConvolution():

    @staticmethod
    def apply_intermediate(x: NDArray[np.float32], prev_kernel: NDArray[np.float32], curr_kernel: NDArray[np.float32], transition_length: int) -> NDArray[np.float32]:
        ksize = max(prev_kernel.size, curr_kernel.size)
        smoothed = np.zeros(x.size + ksize - 1, dtype=np.float32)
        prev_kernel = np.pad(prev_kernel, (0, ksize - prev_kernel.size), mode="constant")
        kernel_padded = np.pad(curr_kernel, (0, ksize - curr_kernel.size), mode="constant")

        transition_length = min(transition_length, smoothed.size)
        for i in range(transition_length):
            alpha = float(i / (transition_length - 1))
            crossed = (1 - alpha) * prev_kernel + alpha * kernel_padded
            smoothed[i:i + ksize] += x[i] * crossed
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
        self.buffer = np.empty(0, dtype=np.float32)
        self.chunk = chunk
        self.pkernel = None
        self.transition_size = int(self.chunk / TRANSITION_FACTOR)

    def process(self, x: NDArray[np.float32], kernel: NDArray[np.float32]) -> NDArray[np.float32]:

        convolution = scipy.signal.fftconvolve(x, kernel, mode="full")
        if self.pkernel is not None:
            convolution = SmoothedConvolution.apply_intermediate(x=x, prev_kernel=self.pkernel, curr_kernel=kernel, transition_length=self.transition_size)

        self.pkernel = kernel

        max_lenght = min(self.buffer.size, convolution.size)
        convolution[:max_lenght] += self.buffer[:max_lenght]
        convolved = convolution[:self.chunk].astype(np.float32)
        self.buffer = convolution[self.chunk:]
        return convolved

class HybriKernels():
    def __init__(self, rir: NDArray[np.float32] | None, hrir: NDArray[np.float32]):
        self.rir = rir
        self.hrir = hrir

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

        self.__cache_positions = {} # implement delete old data to avoid memory problems
        self.__ppoints = Queue()
        self.__hrirs_cache = Queue()

        self.__cache_morph_data = {} # implement delete old data to avoid memory problems
        self.__morph_data = Queue()
        self.__rirs_cache = Queue()

        self.__space_process_is_alive = Event()
        self.__space_process = None

    def close(self) -> None:
        """
        CLOSE HYBRIZONE
        """
        print("[INFO] Close Hybrizone...")

        print("[INFO] Terminate space process")
        self.__space_process_is_alive.set()
        time.sleep(0.1)

        if self.__space_process is not None:
            self.__space_process.join()

        time.sleep(0.1)

        print("[INFO] Free memory")
        self.hrir_builder.close()
        if self.rir_builder is not None:
            self.rir_builder.close()
        
        print("[INFO] Hybrizone closed!")

    def process_frame(self, frame: NDArray[np.float32], kernels: HybriKernels) -> NDArray[np.float32]:
        """
        APPLY RIR AND HRIR

        Parameters
        ----------
        frame : NDArray[np.float32]
            frame to be process
        kernels : HybriKernels
            RIR and HRIR kernel

        Returns
        -------
        NDArray[np.float32]
            frame processed
        """

        mono = frame
        if kernels.rir is not None:
            mono = self.__rir_buffer.process(x=frame, kernel=kernels.rir)

        with ThreadPoolExecutor(max_workers=2) as convolver:
            lc = convolver.submit(self.__hrir_left_buffer.process, mono, kernels.hrir[:, 0])
            lr = convolver.submit(self.__hrir_right_buffer.process, mono, kernels.hrir[:, 1])
            left_hrir = lc.result()
            right_hrir = lr.result()

        convolved = np.column_stack((left_hrir, right_hrir)).astype(np.float32)
        return np.tanh(convolved * SOFT_CLIP_SCALE)

    def __start_space_data_process(self) -> None:
        while not self.__space_process_is_alive.is_set():

            condition = not self.__ppoints.empty()
            if self.rir_builder is not None:
                condition = not self.__ppoints.empty() and not self.__morph_data.empty()

            position = None
            match self.rir_builder:
                case None:
                    if condition:
                        position = self.__ppoints.get()
                        pkey = position._get_hash()

                        check_hrir = self.__cache_positions.get(pkey)

                        hrir = None
                        if check_hrir:
                            hrir = check_hrir

                        else:
                            hrirs = self.query_hrirs(spatial_position=position, n_neighs=self.__params.interpolation_neighs)
                            hrir = self.build_distance_based_hrir(hrirs=hrirs)

                        self.__hrirs_cache.put(hrir)

                case _:
                    if condition:
                        position = self.__ppoints.get()
                        morph = self.__morph_data.get()
                        pkey = position._get_hash()
                        mkey = morph._get_hash()

                        check_hrir, check_rir = self.__cache_positions.get(pkey), self.__cache_morph_data.get(mkey)

                        hrir, rir = None, None
                        if check_hrir and check_rir:
                            hrir = check_hrir
                            rir = check_rir

                        elif check_hrir:
                            hrir = check_hrir
                            rir = self.build_hybrid_space(direction=morph.direction, morph_curve=morph.morph_curve, rho=position.rho)

                        elif check_rir:
                            hrirs = self.query_hrirs(spatial_position=position, n_neighs=self.__params.interpolation_neighs)
                            hrir = self.build_distance_based_hrir(hrirs=hrirs)
                            rir = check_rir

                        else:
                            hrirs = self.query_hrirs(spatial_position=position, n_neighs=self.__params.interpolation_neighs)

                            with ThreadPoolExecutor(max_workers=2) as calculator:
                                hrir_distance_based = calculator.submit(self.build_distance_based_hrir, hrirs)
                                rir_calculated = calculator.submit(self.build_hybrid_space, morph.direction, morph.morph_curve, position.rho)
                                hrir = hrir_distance_based.result()
                                rir = rir_calculated.result()

                        self.__hrirs_cache.put(hrir)
                        self.__rirs_cache.put(rir)

            time.sleep(1 / self.__params.fs)

    def start_space_data_process(self) -> None:
        """
        START RIR AND HRIR PARALLEL CALCULATION PROCESS

        Returns
        -------
            process in which the program calculates RIR and HRIR params-based.
        """

        self.__space_process = Thread(target=self.__start_space_data_process, daemon=True)
        self.__space_process.start()
        print("[INFO] SPACE PROCESS STARTED!")

    def set_position(self, position: PolarPoint) -> None:
        """
        SET CURRENT POSITION

        Parameters
        ----------
        position : PolarPoint
            current position

        """

        if self.__space_process is None:
            print("[INFO] None space process. Run .start_space_data_process() method first!")
            exit(1)

        self.__ppoints.put(position)

    def set_morph_data(self, direction: float, morph_curve: CurveMode) -> None:
        """
        SET CURRENT HYBRID SPACE

        Parameters
        ----------
        direction : float
            morph direction
        morph_curve : CurveMode
            morph curve

        """

        if self.__space_process is None:
            print("[INFO] None space process. Run .start_space_data_process() method first!")
            exit(1)

        self.__morph_data.put(MorpData(direction=direction, morph_curve=morph_curve))


    def get_rir_and_hrir(self) -> HybriKernels|None:
        """
        GET GENERATED RIR AND HRIR

        Returns
        -------
        HybriKernels|None
            kernels or None
        """

        match self.rir_builder:
            case None:
                if self.__hrirs_cache.empty():
                    return None

                hrir = self.__hrirs_cache.get()
                k = HybriKernels(rir=None, hrir=hrir)
                if not isinstance(k, HybriKernels):
                    return None
                return k
            case _:
                if self.__hrirs_cache.empty() or self.__rirs_cache.empty():
                    print("[WARNING] Empty cache!")
                    if not self.__ppoints.empty():
                        print("[WARNING] Searching pos...")
                        k = None
                        time_process = time.time()
                        while k is None:
                            hrir = self.__hrirs_cache.get()
                            rir = self.__rirs_cache.get()
                            k = HybriKernels(rir=rir, hrir=hrir)
                            if not isinstance(k, HybriKernels):
                                k = None
                            if time.time() - time_process >= 0.1:
                                return None
                        return k
                    return None

                hrir = self.__hrirs_cache.get()
                rir = self.__rirs_cache.get()
                k = HybriKernels(rir=rir, hrir=hrir)
                if not isinstance(k, HybriKernels):
                    return None
                return k

        return k

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

    def build_distance_based_hrir(self, hrirs: HInfo) -> NDArray[np.float32]:
        """
        BUILD HRIR FROM HInfo (SELECTED HRIRs)

        Parameters
        ----------
        hrirs : HInfo
            info about selected HRIRs

        Returns
        -------
        NDArray[np.float32]
            builded hrir
        """

        return self.hrir_builder.build_hrir(hrirs_info=hrirs, method=self.__params.build_mode)

    def display_hrir(self, hrir: NDArray[np.float32], title: str) -> None:
        """
        PLOT HRIR ANALYSIS

        Parameters
        ----------
        hrir : NDArray[np.float32]
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

        self.rir_builder.set_rirs(rir1=rir1, rir2=rir2, smooth_factor=smooth_factor)

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

        return self.rir_builder.rirs.attrs["IR-keys"][rir_index]

    def set_smooth_factor(self, smooth_factor: float) -> None:
        """
        SET SMOOTH FACTOR

        Parameters
        ----------
        smooth_factor : float, optional
            spectral envelope smooth factor, by default 0.1
        """

        self.rir_builder.set_smooth_factor(smooth_factor=smooth_factor)

    def build_hybrid_space(self, direction: float, morph_curve: CurveMode, rho: float) -> NDArray[np.float32]:
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
        NDArray[np.float32]
            RIR
        """

        return self.rir_builder.morpha(direction=direction, morph_curve=morph_curve, distance=rho)

    def display_rir(self, rir: int|NDArray[np.float32], title: str|None = None) -> None:
        """
        _summary_

        Parameters
        ----------
        rir : int | NDArray[np.float32]
            RIR defined starting from database index or data array
        title : str | None, optional
            plot title. If rir=int plot title will be the RIR key, by default None
        """

        self.rir_builder.plot_rir(rir=rir, title=title)

    # --- END RIR SECTION ---
