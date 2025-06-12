import numpy as np
from numpy.typing import NDArray
import h5py
import scipy as sp
from concurrent.futures import ThreadPoolExecutor
from hybri_tools import AirData, ISO9613Filter, GeometricAttenuation, ETA, CoordMode, PolarPoint, InterpolationDomain, LRUCache
from hybri_tools import BuildMode, HBuilded, woodworth_itd3d, cross_fade, INTERNAL_KERNEL_TRANSITION, MAX_DISTANCE_TRANSITION, LRU_CAPACITY
# import time


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
        hrirs: NDArray[np.float32]|None = None, 
        hrtfs: NDArray[np.complex64]|None = None, 
        itds: NDArray[np.float32]|None = None, 
        coords: NDArray[np.float64]|None = None,
        distances: NDArray[np.float64]|None = None, 
        target: PolarPoint|None = None, 
        shape: NDArray[np.float32]|None = None, 
        n_neighs: int = 0,
        point_hash: str|None = None
    ) -> None:
        
        self.hrirs = hrirs
        self.hrtfs = hrtfs
        self.itds = itds
        self.coords = coords
        self.distances = distances
        self.target = target
        self.shape = shape
        self.n_neighs = n_neighs
        self.point_hash = point_hash

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
        
        self.__prev_dist_hrir = None
        self.__prev_distance = None
        
        self.__cache_hrir_builded = LRUCache[HBuilded](capacity=LRU_CAPACITY)
        
    def close(self) -> None:
        self.dataset.close_data()
    
    def set_air_conditions(self, air_data: AirData) -> None:
        self.iso9613 = ISO9613Filter(air_data=air_data, fs=self.fs)
        self.db_attenuation = self.iso9613.get_attenuation_air_absorption()
    
    def __interpolated_hrir(self, hrirs_info: HInfo, method: BuildMode, mode: InterpolationDomain) -> HBuilded:
        h = hrirs_info.hrirs if mode == InterpolationDomain.TIME else hrirs_info.hrtfs
        
        interpolated_freq = None
        interpolated_time = None
        interpolated_itd = None
        
        check_itd_distance = True if hrirs_info.target.rho < self.dataset.get_source_distance() else False
        gfac = hrirs_info.target.rho - self.source_distance
        gfac = gfac if gfac > 0 else 0
        gain = -self.db_attenuation * gfac

        min_arg = np.argmin(hrirs_info.distances)
        if hrirs_info.distances[min_arg] < 1e-5:
            return HBuilded(hrir=hrirs_info.hrirs[min_arg], itd=hrirs_info.itds[min_arg], gain=gain)
        
        match method:
                        
            case BuildMode.INVERSE_DISTANCE | BuildMode.LINEAR_INVERSE_DISTANCE:
                w = 1 / hrirs_info.distances ** 2 if method == BuildMode.INVERSE_DISTANCE else 1 / hrirs_info.distances
                w /= np.sum(w)
                
                ref_itds = hrirs_info.itds[:2]
                itds_ref_std = np.std(ref_itds)
                itds_temp = hrirs_info.itds
                if np.std(hrirs_info.itds) > 3 * itds_ref_std:
                    itds_temp = ref_itds
                    w = 1 / hrirs_info.distances[:2] ** 2 if method == BuildMode.INVERSE_DISTANCE else 1 / hrirs_info.distances[:2]
                    w /= np.sum(w)
                
                if check_itd_distance:
                    interpolated_itd = woodworth_itd3d(point=hrirs_info.target)
                else:
                    interpolated_itd = np.sum([w[i] * itds_temp[i] for i in range(len(itds_temp))])
                
                match mode:
                    case InterpolationDomain.TIME:
                        interpolated_time = np.sum([w[i] * h[i] for i in range(len(itds_temp))], axis=0)
                    case InterpolationDomain.FREQUENCY:
                        interpolated_freq = np.sum(
                            [w[i] * h[i]["fft"] for i in range(len(itds_temp))], 
                            axis=0
                        )
            
            case BuildMode.LINEAR:
                    d1, d2 = hrirs_info.distances[0], hrirs_info.distances[1]
                    alpha = d1 / (d1 + d2)
                    # print(alpha)
                    
                    if check_itd_distance:
                        interpolated_itd = woodworth_itd3d(point=hrirs_info.target)
                    else:
                        interpolated_itd = (1 - alpha) * hrirs_info.itds[0] + alpha * hrirs_info.itds[1]
                    
                    match mode:
                        case InterpolationDomain.TIME:
                            interpolated_time = (1 - alpha) * h[0] + alpha * h[1]
                        case InterpolationDomain.FREQUENCY:
                            interpolated_freq = (1 - alpha) * h[0]["fft"] + alpha * h[1]["fft"]
            
            case BuildMode.SPHERICAL:
                target = hrirs_info.target.get_cartesian(mode=self.mode)
                target = np.array([target.x, target.y, target.z], dtype=np.float64)
                
                tnorm = np.linalg.norm(target)
                h1norm = np.linalg.norm(hrirs_info.coords[0])
                h2norm = np.linalg.norm(hrirs_info.coords[1])
                
                tar = target / tnorm
                h1 = hrirs_info.coords[0] / h1norm
                h2 = hrirs_info.coords[1] / h2norm
                
                omega = np.arccos(np.clip(np.dot(h1, h2), -1.0, 1.0))
                
                if omega == 0.0:
                    a = 0.0
                    b = 1.0
                else:
                    omega1 = np.arccos(np.clip(np.dot(tar, h1), -1.0, 1.0))
                    omega2 = np.arccos(np.clip(np.dot(tar, h2), -1.0, 1.0))
                    sin_omega = np.sin(omega)

                    alpha = omega1 / (omega1 + omega2)
                    a = np.sin((1 - alpha) * omega) / sin_omega
                    b = np.sin(alpha * omega) / sin_omega
                
                if check_itd_distance:
                    interpolated_itd = woodworth_itd3d(point=hrirs_info.target)
                else:
                    interpolated_itd = a * hrirs_info.itds[0] + b * hrirs_info.itds[1]
                
                match mode:
                    case InterpolationDomain.TIME:
                        interpolated_time = a * h[0] + b * h[1]
                    case InterpolationDomain.FREQUENCY:
                        mag = a * h[0]["mag"] + b * h[1]["mag"]
                        phase = a * np.unwrap(h[0]["angle"], axis=0) + b * np.unwrap(h[1]["angle"], axis=0)
                        interpolated_freq = mag * np.exp(1j * phase)
        
        interpolated_itd = -interpolated_itd
        freqs = np.fft.rfftfreq(self.hrir_shape[0], d=1 / self.fs)
        
        y = interpolated_freq if mode == InterpolationDomain.FREQUENCY else interpolated_time
        left, right = y[:, 0], y[:, 1]
        
        result = None
        if mode == InterpolationDomain.FREQUENCY:
            if interpolated_itd > 0:
                right *= np.exp(-2j * np.pi * freqs * interpolated_itd)
            else:
                left *= np.exp(-2j * np.pi * freqs * abs(interpolated_itd))

            ifft_left = np.fft.irfft(left).astype(np.float32)
            ifft_right = np.fft.irfft(right).astype(np.float32)
            result = np.column_stack((ifft_left, ifft_right)).astype(np.float32)

        else:
            samples_delay = int(round(interpolated_itd * self.fs))
            if interpolated_itd > 0:
                right = np.roll(right, -samples_delay, axis=0)
                right[:samples_delay] = 0.0
            else:
                left = np.roll(left, samples_delay, axis=0)
                left[:abs(samples_delay)] = 0.0
            
            result = np.column_stack((left, right)).astype(np.float32)
        
        return HBuilded(hrir=result, itd=interpolated_itd, gain=gain)
    
    def __distance_based_hrir(self, hrir: NDArray[np.float64], rho: float) -> NDArray[np.float64]:
        factor = self.geometric_attenuation.calculate_geometric_attenuation(source_distance=self.source_distance, distance=rho)
        factor = factor if factor <= 1.0 else 1.0
        
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
        
        neighs = neighs if neighs >= 2 else 2
        
        point.rho = max(point.rho, ETA)
        point_hash = point._get_hash()
        
        if self.__cache_hrir_builded.get(key=point_hash) is not None:
            return HInfo(point_hash=point_hash)
            
        cart = point.get_cartesian(mode=self.mode)
        distances, indices = self.dataset.neighs_finder.query([cart.x, cart.y, cart.z], k=neighs)
        
        hinfo = HInfo(
            hrirs=np.empty((neighs, self.hrir_shape[0], self.hrir_shape[1])),
            hrtfs=[],
            itds=np.empty(neighs, dtype=np.float32),
            coords=np.zeros((neighs, 3)),
            distances=np.zeros(neighs, dtype=np.float64),
            target=point,
            shape=self.hrir_shape,
            n_neighs=neighs
        )
        
        # print("------")
        for i, min_index in enumerate(indices):
            coord = self.dataset.get_polar_reference(index=min_index)
            azim, elev = coord
            # print(f"[DEBUG] phi: {elev} | theta: {azim}")
                
            key = f"{int(elev)}_{int(azim)}"
            hinfo.hrirs[i, :, :] = self.dataset.get_hrir(key=key)[:]
            hinfo.hrtfs.append(self.dataset.get_hrtf(key=key))
            hinfo.itds[i] = self.dataset.get_itd(key=key)
            hinfo.coords[i] = self.dataset.get_cartesian_reference(index=min_index)
            hinfo.distances = distances
        # print("------")
        
        return hinfo
    
    def build_hrir(self, hrirs_info: HInfo, method: BuildMode) -> HBuilded:
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
        HBuilded
            
        """
        
        if hrirs_info.point_hash is not None:
            return self.__cache_hrir_builded.get(key=hrirs_info.point_hash)
        
        interpolated = self.__interpolated_hrir(hrirs_info=hrirs_info, method=method, mode=self.interp_domain)
        dhrir_ = self.__distance_based_hrir(hrir=interpolated.hrir, rho=hrirs_info.target.rho) 
        dhrir = dhrir_
        
        if self.__prev_distance is not None and self.__prev_distance != hrirs_info.target.rho:
            d = abs(hrirs_info.target.rho - self.__prev_distance)
            if d > MAX_DISTANCE_TRANSITION:
                tlength = int(INTERNAL_KERNEL_TRANSITION * self.fs)
                dhrir = cross_fade(k1=self.__prev_dist_hrir, k2=dhrir_, tlength=tlength)
            
        self.__prev_dist_hrir = dhrir_
        self.__prev_distance = hrirs_info.target.rho
        point_hash = hrirs_info.target._get_hash()
        interpolated.hrir = dhrir_
        self.__cache_hrir_builded.put(key=point_hash, value=interpolated)
        interpolated.hrir = dhrir
        return interpolated