import numpy as np
from numpy.typing import NDArray
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy as sp
from concurrent.futures import ThreadPoolExecutor
from hybri_tools import AirData, ISO9613Filter, GeometricAttenuation, GAMMA, ETA, CoordMode, PolarPoint, InterpolationDomain, BuildMode

class HrirHDFData():
    def __init__(self, dataset_path: str, coord_mode: CoordMode):
        self.data = h5py.File(dataset_path, "r")
        self.coord_mode = coord_mode
        self.neighs_finder = self.__load_ktree(mode=self.coord_mode)
    
    def get_hrir(self, key: str) -> NDArray[np.float64]:
        return self.data["hrir"][key]
    
    def get_hrtf(self, key: str) -> NDArray[np.float64]:
        return self.data["hrir_fft"][key]
    
    def get_polar_reference(self, index: int) -> NDArray[np.int16]:
        return self.data["polar_index"][index]
    
    def get_itd(self, key: str) -> np.float64:
        return self.data["hrir_itd"][key][()]
    
    def get_sample_rate(self) -> np.float64:
        return self.data.attrs["fs"]
    
    def get_source_distance(self) -> np.float64:
        return self.data.attrs["source_distance"]
    
    def get_cartesian_reference(self, index: int) -> NDArray[np.float64]:
        match self.coord_mode:
            case CoordMode.REGULAR:
                return self.data["regular_coords"][index]
            case CoordMode.INTERAURAL:
                return self.data["interaural_coords"][index]    
    
    def get_shape(self) -> NDArray[np.int16]:
        return self.data.attrs["hrir_shape"]
    
    def close_data(self) -> None:
        self.data.close()

    def __load_ktree(self, mode: CoordMode) -> sp.spatial.cKDTree:
        coords = self.data["regular_coords"][:] if mode == CoordMode.REGULAR else self.data["interaural_coords"][:]
        return sp.spatial.cKDTree(data=coords)


class HInfo():
    def __init__(
        self, 
        hrirs: NDArray[np.float32], 
        hrtfs: NDArray[np.complex64], 
        itds: NDArray[np.float32], 
        coords: NDArray[np.int16],
        distances: NDArray[np.float64], 
        target: PolarPoint, 
        shape: NDArray[np.float32], 
        n_neighs: int
    ) -> None:
        
        self.hrirs = hrirs
        self.hrtfs = hrtfs
        self.itds = itds
        self.coords = coords
        self.distances = distances
        self.target = target
        self.shape = shape
        self.n_neighs = n_neighs

class HRIRBuilder():
    
    def __init__(self, hrir_database: str, mode: CoordMode, interp_domain: InterpolationDomain):
        self.dataset = HrirHDFData(dataset_path=hrir_database, coord_mode=mode)
        self.mode = mode
        self.hrir_shape = self.dataset.get_shape()
        self.fs = self.dataset.get_sample_rate()
        self.source_distance = self.dataset.get_source_distance()
        self.geometric_attenuation = GeometricAttenuation(fs=self.fs, channels=2)
        self.interp_domain = interp_domain
        self.iso9613 = None
        self.db_attenuation = None
        
        self.__att_factor = None
        self.__p = None
        self.__air_conditions = None
    
    def close(self) -> None:
        self.dataset.close_data()
    
    def set_air_conditions(self, air_data: AirData) -> None:
        self.__air_conditions = air_data
        self.iso9613 = ISO9613Filter(air_data=air_data, fs=self.fs)
        self.db_attenuation = self.iso9613.get_attenuation_air_absorption()
    
    def __interpolated_hrir(self, hrirs_info: HInfo, method: BuildMode, mode: InterpolationDomain) -> NDArray[np.float32]:
        h = hrirs_info.hrirs if mode == InterpolationDomain.TIME else hrirs_info.hrtfs
        
        interpolated_freq = None
        interpolated_time = None
        interpolated_itd = None

        min_arg = np.argmin(hrirs_info.distances)
        if hrirs_info.distances[min_arg] < 1e-6:
            return hrirs_info.hrirs[min_arg]
        
        match method:
                        
            case BuildMode.INVERSE_DISTANCE | BuildMode.LINEAR_INVERSE_DISTANCE:
                w = 1 / hrirs_info.distance ** 2 if method == BuildMode.INVERSE_DISTANCE else 1 / hrirs_info.distance
                w /= np.sum(w)
                
                interpolated_itd = np.sum([w[i] * hrirs_info.itds[i] for i in range(hrirs_info.n_neighs)])
                
                match mode:
                    case InterpolationDomain.TIME:
                        interpolated_time = np.sum([w[i] * h[i] for i in range(hrirs_info.n_neighs)], axis=0)
                    case InterpolationDomain.FREQUENCY:
                        interpolated_freq = np.sum([w[i] * (h[i]["mag"] * np.exp(1j * np.unwrap(h[i]["angle"], axis=0))) for i in range(hrirs_info.n_neighs)], axis=0)
            
            case BuildMode.LINEAR:
                    i, j = 0, 1
                    d1, d2 = hrirs_info.distances[i], hrirs_info.distances[j]
                    alpha = d1 / (d1 + d2)
                    # print(alpha)
                    
                    interpolated_itd = (1 - alpha) * hrirs_info.itds[i] + alpha * hrirs_info.itds[j]
                    
                    match mode:
                        case InterpolationDomain.TIME:
                            interpolated_time = (1 - alpha) * h[i] + alpha * h[j]
                        case InterpolationDomain.FREQUENCY:
                            hi = h[i]["mag"] * np.exp(1j * np.unwrap(h[i]["angle"], axis=0))
                            hj = h[j]["mag"] * np.exp(1j * np.unwrap(h[j]["angle"], axis=0))
                            interpolated_freq = (1 - alpha) * hi + alpha * hj
        
        freqs = np.fft.rfftfreq(self.hrir_shape[0], d=1 / self.fs)
        
        y = interpolated_freq if mode == InterpolationDomain.FREQUENCY else interpolated_time
        left, right = y[:, 0], y[:, 1]
        
        result = None
        if mode == InterpolationDomain.FREQUENCY:
            if interpolated_itd > 0:
                right *= np.exp(-2j * np.pi * freqs * interpolated_itd)
            else:
                left *= np.exp(-2j * np.pi * freqs * abs(interpolated_itd))

            ifft_left = np.fft.irfft(left, n=self.hrir_shape[0], axis=0).astype(np.float32)
            ifft_right = np.fft.irfft(right, n=self.hrir_shape[0], axis=0).astype(np.float32)
            result = np.column_stack((ifft_left, ifft_right)).astype(np.float32)

        else:
            samples_delay = int(round(interpolated_itd * self.fs))
            if interpolated_itd > 0:
                right = np.roll(interpolated_time, -samples_delay, axis=0)
                right[:samples_delay] = 0.0
            else:
                left = np.roll(interpolated_time, samples_delay, axis=0)
                left[:abs(samples_delay)] = 0.0
            
            result = np.column_stack((left, right)).astype(np.float32)
        
        return result
    
        
    def __distance_based_hrir(self, hrir: NDArray[np.float64], rho: float) -> NDArray[np.float64]:
        factor = self.geometric_attenuation.calculate_geometric_attenuation(source_distance=self.source_distance, distance=rho)
        self.__att_factor = factor
        
        if factor < 1.0:
            with ThreadPoolExecutor(max_workers=2) as delayer:
                lc = delayer.submit(self.geometric_attenuation.apply_fractional_delay, signal=hrir[:, 0], distance=rho, channel=0)
                rc = delayer.submit(self.geometric_attenuation.apply_fractional_delay, signal=hrir[:, 1], distance=rho, channel=1)
                hrir[:, 0] = lc.result()
                hrir[:, 1] = rc.result()
                
        filtered = self.iso9613.air_absorption_filter(frame=hrir, alpha_absortion=self.db_attenuation, distance=rho - max(self.source_distance, ETA)) * factor
        
        return filtered
    
    def prepare_hrirs(self, point: PolarPoint, neighs: int) -> HInfo:
        """
        PREPARE HRIRs

        Parameters
        ----------
        point : PolarPoint
            target point in polar coordinates (see PolarPoint, [rho, phi, theta])
        neighs : int
            how many neighbors (for interpolation)

        Returns
        -------
        HInfo
        
        """
        
        point.rho = max(point.rho, ETA)
        self.__p = point
        
        cart = point.get_cartesian(mode=self.mode)
        distances, indices = self.dataset.neighs_finder.query([cart.x, cart.y, cart.z], k=neighs)
        
        hinfo = HInfo(
            hrirs=np.empty((neighs, self.hrir_shape[0], self.hrir_shape[1])),
            hrtfs=[],
            itds=np.empty(neighs, dtype=np.float32),
            coords=np.zeros((neighs, 2)),
            distances=np.zeros(neighs, dtype=np.float64),
            target=point,
            shape=self.hrir_shape,
            n_neighs=neighs
        )
        
        for i, min_index in enumerate(indices):
            coord = self.dataset.get_polar_reference(index=min_index)
            elev, azim = None, None
            
            if self.mode == CoordMode.INTERAURAL:
                c = self.dataset.get_cartesian_reference(index=min_index)
                x, y, z = c[0], c[1], c[2]
                elev_temp = np.arctan2(z, y)
                azim_temp = np.arcsin(x)

                elev = np.rad2deg(elev_temp)
                azim = np.rad2deg(azim_temp) if y >= 0.0 else np.rad2deg(azim_temp) + 180
            else:
                azim, elev = coord
                
            key = f"{int(elev)}:{int(azim)}"
            hinfo.hrirs[i, :, :] = self.dataset.get_hrir(key=key)[:]
            hinfo.hrtfs.append(self.dataset.get_hrtf(key=key))
            hinfo.itds[i] = self.dataset.get_itd(key=key)
            hinfo.coords[i] = np.array([int(elev), int(azim)])
            hinfo.distances = distances
            
        return hinfo
    
    def build_hrir(self, hrirs_info: HInfo, method: BuildMode) -> NDArray[np.float64]:
        """
        GENERATE INTERPOLATED DISTANCE-BASED HRIR

        Parameters
        ----------
        hrirs_info : HInfo
            generated HRIRs info from self.prepare_hrirs method
        method : BuildMode
            interpolation mode (see BuildMode)
        hifreq : float, optional
            air absorption filter cut off frequency, by default 18000.0
        c : float, optional
            speed of sound propagation, by default 343.3

        Returns
        -------
        NDArray[np.float64]
            interpolated distance based hrir left and right channel
        """
        
        if hrirs_info.n_neighs == 1: 
            return self.__distance_based_hrir(hrir=hrirs_info.hrirs, rho=hrirs_info.target.rho)
        
        interpolated = self.__interpolated_hrir(hrirs_info=hrirs_info, method=method, mode=self.interp_domain)
        return self.__distance_based_hrir(hrir=interpolated, rho=hrirs_info.target.rho)
    
    def plot_hrir(self, data: NDArray[np.float64], title: str) -> None:
        sr = self.fs
        f = np.fft.rfftfreq(data.shape[0], d = 1 / sr)
        ps = np.abs(np.fft.rfft(data, axis=0))
        t = np.arange(data.shape[0]) / sr
        
        rho = np.round(self.__p.rho, decimals=3)
        phi = np.round(self.__p.phi, decimals=3)
        theta = np.round(self.__p.theta, decimals=3)
        
        d = rho - self.source_distance
        d = d if d > 0 else 0
        db_attenuation = -self.db_attenuation * d

        _ = plt.figure(figsize=(12, 9))
        # fig.subplots_adjust(hspace=10)
        gs = gridspec.GridSpec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1])
        ax0 = plt.subplot(gs[:2, 0])
        ax1 = plt.subplot(gs[2:4, 0])
        ax2 = plt.subplot(gs[4:, 0])
        ax3 = plt.subplot(gs[:3, 1])
        ax4 = plt.subplot(gs[3:, 1])
        
        temp = self.__air_conditions.kelvin - 273.15
        hum = self.__air_conditions.humidity * 100
        pres = self.__air_conditions.pressure * 101325.0
        phi = ((np.rad2deg(phi) + 90) % 180) - 90
        phi = np.round(phi, decimals=3)
        theta = np.round(np.rad2deg(theta) % 360, decimals=3)
        
        plt.suptitle(f'{title}' "\n" rf'$T = {temp} \degree C$, $RH = {hum} \%$, $P = {pres} Pa$' "\n" rf'$(\rho, \phi, \theta) = {rho, phi, theta}$, $\varrho_0 = {self.source_distance}$, $\varrho = {rho}$, $\gamma(\varrho) = {np.round(self.__att_factor, decimals=3)}$, $y = {GAMMA}$, ' r'$\varrho_{min} = $' rf'${ETA}$')
        
        ax0.plot(f, ps[:, 0], c="k", lw=0.5)
        ax0.set_title("LEFT CH. POWER SPECTRUM")
        ax0.set_xlabel("Freq.")
        ax0.set_ylabel("Mag.")
        ax1.plot(f, ps[:, 1], c="k", lw=0.5)
        ax1.set_title("RIGHT CH. POWER SPECTRUM")
        ax1.set_xlabel("Freq.")
        ax1.set_ylabel("Mag.")
       
        ax2.plot(self.iso9613._frequencies, db_attenuation, c="k", lw=0.5)
        ax2.set_title("IMPOSED ISO 9613-1")
        ax2.set_xlabel("Freq.")
        ax2.set_ylabel("dB")
        
        ax3.plot(t, data[:, 0], c="k", lw=0.7)
        ax3.set_title("LEFT CH. WAVEFORM")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Amp.")
        ax4.plot(t, data[:, 1], c="k", lw=0.7)
        ax4.set_title("RIGHT CH. WAVEFORM")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Amp.")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
    