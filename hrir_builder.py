from common_modules import np, h5py, NDArray, Enum, plt, gridspec
import scipy as sp
import hashlib

class CoordMode(Enum):
    INTERAURAL = 0
    REGULAR = 1

class AngleMode(Enum):
    RADIANS = 0
    DEGREE = 1

class BuildMode(Enum):
    BILINEAR = 0
    INVERSE_DISTANCE = 1
    LINEAR_INVERSE_DISTANCE = 2
    LINEAR = 3
    HERMITE = 4
    
class InterpolationDomain(Enum):
    TIME = 0
    FREQUENCY = 1

class CartesianPoint():
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.z = z
        self.y = y

class AirData():
    def __init__(self, temperature: float, humidity: float, pressure: float = 101325):
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

class HInfo():
    def __init__(
        self, hrirs: NDArray[np.float32], 
        hffts: NDArray[np.complex64], 
        itds: NDArray[np.float32], 
        coords: NDArray[np.int16], 
        target: PolarPoint, 
        shape: NDArray[np.float32], 
        n_neighs: int
    ) -> None:
        
        self.hrirs = hrirs
        self.hffts = hffts
        self.itds = itds
        self.coords = coords
        self.target = target
        self.shape = shape
        self.n_neighs = n_neighs

class HRIRBuilder():
    
    # DYNAMIC DISTANCE FILTER 
    NFREQS = 256
    
    # DISTANCE PERCPETION FACTORS
    GAMMA = 0.36 # exponent in distance perception
    ETA = 0.01 # minimum distance threshold
    
    def __init__(self, hrir_database: str, mode: CoordMode, interp_domain: InterpolationDomain):
        self.h = h5py.File(hrir_database, "r")
        self.mode = mode
        self.coordinates = self.h["regular_coords"][:] if self.mode == CoordMode.REGULAR else self.h["interaural_coords"][:]
        self.polar_reference = self.h["polar_index"][:]
        self.hrirs = self.h["hrir"]
        self.hffts = self.h["hrir_fft"]
        self.itds = self.h["hrir_itd"]
        self.hrir_shape = self.h.attrs["hrir_shape"]
        self.fs = self.h.attrs["fs"]
        self.source_distance = self.h.attrs["source_distance"]
        self.interp_domain = interp_domain
        self.tree = self.__load_tree()
        self.__frequencies = np.linspace(0, self.fs / 2, self.NFREQS)
        self.db_attenuation = None
        self.__att_factor = None
        self.__p = None
        self.__air_conditions = None
    
    def close(self) -> None:
        self.h.close()
    
    def set_air_conditions(self, air_data: AirData) -> None:
        self.__air_conditions = air_data
        self.db_attenuation = self.__air_absorption(kelvin=air_data.kelvin, humidity=air_data.humidity, p_atm=air_data.pressure)
    
    def __load_tree(self) -> sp.spatial.cKDTree:
        return sp.spatial.cKDTree(data=self.coordinates)
    
    def __calculate_distance(self, hrirs, coords, target_ele, target_azi) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
        c = np.column_stack([coords[:, 0], coords[:, 1]])
        t = np.array([target_ele, target_azi]).reshape(1, -1)
        distance = sp.spatial.distance_matrix(c, t).flatten()
        if np.min(distance) < 1e-6:
            closest = np.argmin(distance)
            return None, hrirs.hrirs[closest]
        return distance, None
    
    def __interpolated_hrir(self, hrirs: HInfo, method: BuildMode, mode: InterpolationDomain) -> NDArray[np.float32]:
        coords = hrirs.coords
        min_azi, max_azi = np.min(coords[:, 1]), np.max(coords[:, 1])
        min_ele, max_ele = np.min(coords[:, 0]), np.max(coords[:, 0])
        
        if method in [BuildMode.BILINEAR, BuildMode.HERMITE] and hrirs.n_neighs != 4:
            print("[ERROR] Methods BILINEAR and HERMITE requires 4 neighs!")
            exit(1)
            
        target_azi = hrirs.target.theta
        target_ele = hrirs.target.phi
        
        h = None
        
        match mode:
            case InterpolationDomain.TIME:
                h = hrirs.hrirs
            case InterpolationDomain.FREQUENCY:
                h = hrirs.hffts
        
        interpolated_freq = None
        interpolated_time = None
        interpolated_itd = None

        match method:
            case BuildMode.BILINEAR:
                den = (max_azi - min_azi) * (max_ele - min_ele)
                w1 = ((max_azi - target_azi) * (max_ele - target_ele)) / den
                w2 = ((target_azi - min_azi) * (max_ele - target_ele)) / den
                w3 = ((max_azi - target_azi) * (target_ele - min_ele)) / den
                w4 = ((target_azi - min_azi) * (target_ele - min_ele)) / den
                
                interpolated_itd = w1 * hrirs.itds[0] + w2 * hrirs.itds[1] + w3 * hrirs.itds[2] + w4 * hrirs.itds[3]
                
                match mode:
                    case InterpolationDomain.TIME:
                        interpolated_time = w1 * h[0] + w2 * h[1] + w3 * h[2] + w4 * h[3]
                    case InterpolationDomain.FREQUENCY:
                        m1, m2, m3, m4 = h[0]["mag"], h[1]["mag"], h[2]["mag"], h[3]["mag"]
                        p1, p2, p3, p4 = np.unwrap(h[0]["angle"], axis=0), np.unwrap(h[1]["angle"], axis=0), np.unwrap(h[2]["angle"], axis=0), np.unwrap(h[3]["angle"], axis=0)
                        interpolated_freq = (w1 * (m1 * np.exp(1j * p1)) + w2 * (m2 * np.exp(1j * p2)) + w3 * (m3 * np.exp(1j * p3)) + w4 * (m4 * np.exp(1j * p4)))
                        
            case BuildMode.INVERSE_DISTANCE | BuildMode.LINEAR_INVERSE_DISTANCE | BuildMode.LINEAR:
                distance, hsingular = self.__calculate_distance(hrirs=hrirs, coords=coords, target_ele=target_ele, target_azi=target_azi)
                if distance is None:
                    return hsingular
                
                match method:
                    case BuildMode.INVERSE_DISTANCE | BuildMode.LINEAR_INVERSE_DISTANCE:
                        w = 1 / distance ** 2 if method == BuildMode.INVERSE_DISTANCE else 1 / distance
                        w /= np.sum(w)
                        
                        interpolated_itd = np.sum([w[i] * hrirs.itds[i] for i in range(hrirs.n_neighs)])
                        
                        match mode:
                            case InterpolationDomain.TIME:
                                interpolated_time = np.sum([w[i] * h[i] for i in range(hrirs.n_neighs)], axis=0)
                            case InterpolationDomain.FREQUENCY:
                                interpolated_freq = np.sum([w[i] * (h[i]["mag"] * np.exp(1j * np.unwrap(h[i]["angle"], axis=0))) for i in range(hrirs.n_neighs)], axis=0)
                    
                    case BuildMode.LINEAR:
                            indexes = np.argsort(distance)
                            # print(indexes)
                            i, j = indexes[:2]
                            d1, d2 = distance[i], distance[j]
                            alpha = d1 / (d1 + d2)
                            # print(alpha)
                            
                            interpolated_itd = (1 - alpha) * hrirs.itds[i] + alpha * hrirs.itds[j]
                            
                            match mode:
                                case InterpolationDomain.TIME:
                                    interpolated_time = (1 - alpha) * h[i] + alpha * h[j]
                                case InterpolationDomain.FREQUENCY:
                                    hi = h[i]["mag"] * np.exp(1j * np.unwrap(h[i]["angle"], axis=0))
                                    hj = h[j]["mag"] * np.exp(1j * np.unwrap(h[j]["angle"], axis=0))
                                    interpolated_freq = (1 - alpha) * hi + alpha * hj
            
            case BuildMode.HERMITE:
                distance, hsingular = self.__calculate_distance(hrirs=hrirs, coords=coords, target_ele=target_ele, target_azi=target_azi)
                if distance is None:
                    return hsingular
                
                indexes = np.argsort(distance)
                i0, i1, i2, i3 = indexes
                d1, d2 = distance[i1], distance[i2]
                mu = d1 + d2 / np.sum(distance)
                
                itd0 = -0.5 * hrirs.itds[i0] + 1.5 * hrirs.itds[i1] - 1.5 * hrirs.itds[i2] + 0.5 * hrirs.itds[i3]
                itd1 = hrirs.itds[i0] - 2.5 * hrirs.itds[i1] + 2.0 * hrirs.itds[i2] - 0.5 * hrirs.itds[i3]
                itd2 = -0.5 * hrirs.itds[i0] + 0.5 * hrirs.itds[i2]
                itd3 = hrirs.itds[i1]

                mu_itd = hrirs.itds[i1] + hrirs.itds[i2] / np.sum(hrirs.itds) 
                interpolated_itd = itd0 * pow(mu_itd, 3) + itd1 * pow(mu_itd, 2) + itd2 + itd3
                
                match mode:
                    case InterpolationDomain.TIME:
                        a0 = -0.5 * h[i0] + 1.5 * h[i1] - 1.5 * h[i2] + 0.5 * h[i3]
                        a1 = h[i0] - 2.5 * h[i1] + 2.0 * h[i2] - 0.5 * h[i3]
                        a2 = -0.5 * h[i0] + 0.5 * h[i2]
                        a3 = h[i1]
                        
                        interpolated_time = a0 * pow(mu, 3) + a1 * pow(mu, 2) + a2 + a3
                        
                    case InterpolationDomain.FREQUENCY:

                        m1, m2, m3, m4 = h[i0]["mag"], h[i1]["mag"], h[i2]["mag"], h[i3]["mag"]
                        p1, p2, p3, p4 = np.unwrap(h[i0]["angle"], axis=0), np.unwrap(h[i1]["angle"], axis=0), np.unwrap(h[i2]["angle"], axis=0), np.unwrap(h[i3]["angle"], axis=0)

                        a0 = -0.5 * (m1 * np.exp(1j * p1)) + 1.5 * (m2 * np.exp(1j * p2)) - 1.5 * (m3 * np.exp(1j * p3)) + 0.5 * (m4 * np.exp(1j * p4))
                        a1 = (m1 * np.exp(1j * p1)) - 2.5 * (m2 * np.exp(1j * p2)) + 2.0 * (m3 * np.exp(1j * p3)) - 0.5 * (m4 * np.exp(1j * p4))
                        a2 = -0.5 * (m1 * np.exp(1j * p1)) + 0.5 * (m3 * np.exp(1j * p3))
                        a3 = m2 * np.exp(1j * p2)

                        interpolated_freq = a0 * pow(mu, 3) + a1 * pow(mu, 2) + a2 + a3

            case _:
                print("[ERROR] Method not allowed!")
                exit(1)
        
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
    
    def __air_absorption(self, kelvin: float, humidity: float, p_atm: float) -> NDArray[np.float32]:
        h = humidity * 10 ** ((-6.8346 * (273.16 / kelvin) ** 1.261) + 4.6151)
        
        # Frequenze di rilassamento per ossigeno e azoto (Hz)
        f_rO = p_atm * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
        f_rN = p_atm * (9.0 + 280.0 * h * np.exp(-4.17 * ((kelvin / 293.0) - 1)))
        
        # Calcolo del coefficiente di assorbimento in dB/m
        freq_kHz = self.__frequencies / 1000.0  # conversione in kHz
        
        # Termine per l'assorbimento classico (viscosità e conduzione termica)
        alpha_classical = 1.84e-11 * (p_atm) ** (-1) * np.sqrt(kelvin / 293.15) * freq_kHz ** 2
        
        # Termine per il rilassamento dell'ossigeno
        alpha_oxygen = 0.01275 * np.exp(-2239.1 / kelvin) * freq_kHz ** 2 / (f_rO / 1000.0 + (freq_kHz ** 2 / f_rO / 1000.0))
        
        # Termine per il rilassamento dell'azoto
        alpha_nitrogen = 0.1068 * np.exp(-3352.0 / kelvin) * freq_kHz ** 2 / (f_rN / 1000.0 + (freq_kHz ** 2 / f_rN / 1000.0))
        
        # Coefficiente di assorbimento totale (in dB/m)
        alpha = alpha_classical + alpha_oxygen + alpha_nitrogen
        
        return alpha.astype(np.float32)
    
    def __multiband_fft_filter(self, frame: NDArray[np.float32], attenuation: NDArray[np.float32], fs: float):
        n = len(frame)
        fpoints = self.__frequencies / (fs / 2) # normalized
        gain_interpolator = sp.interpolate.interp1d(fpoints, attenuation, bounds_error=False, fill_value=(attenuation[0], attenuation[-1]))
        fft_freqs = np.linspace(0, 1, n // 2 + 1)
        fresp = gain_interpolator(fft_freqs)
        fresp = fresp[:, None].astype(np.complex64)
        fft = np.fft.rfft(frame, axis=0)
        filtered_fft = fft * fresp
        filtered = np.fft.irfft(filtered_fft, n=n, axis=0)
        return filtered
    
    def __air_absorption_filter(self, frame: NDArray[np.float64], distance: float) -> NDArray[np.float64]:
        # Filtro basato su ISO 9613-1 (see __air_absorption)
        if self.db_attenuation is None:
            print("[ERROR] Air Conditions must be defined (use .set_air_conditions() method first)!")
        relative_distance = distance if distance > 0 else 0
        db_attenuation = self.db_attenuation * relative_distance
        gain = 10 ** (-db_attenuation / 20.0)
        gain[-1] = 0.0
        filtered = self.__multiband_fft_filter(frame=frame, attenuation=gain, fs=self.fs)
        return filtered
        
    def __distance_based_hrir(self, hrir: NDArray[np.float64], rho: float, c: float) -> NDArray[np.float64]:
        original_distance = max(self.source_distance, 1e-6)
        fs = self.fs
        factor = (original_distance / rho) ** self.GAMMA # il fattore di attenuazione è in rapporto con la distanza originale in modo tale da lasciare invariata l'ampiezza alla distanza originale
        self.__att_factor = factor
        
        delayed_samples = rho * fs / c
        int_delay = int(np.floor(delayed_samples))
        frac_delay = delayed_samples - int_delay
        
        h = np.pad(hrir, ((int_delay, 0), (0, 0)), mode="constant")
        
        if frac_delay != 0:
            x = np.arange(h.shape[0])
            interpolator = sp.interpolate.interp1d(x, h, axis=0, kind='linear', fill_value="extrapolate")
            h = interpolator(x + frac_delay)
        
        filtered = self.__air_absorption_filter(frame=h, distance=rho - original_distance) * factor
        # print(factor)
        return filtered * factor
    
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
        
        point.rho = max(point.rho, self.ETA)
        self.__p = point
        
        # if target exists in polar references
        check = np.rad2deg(np.array([point.theta, point.phi]))
        check = np.ceil(check).astype(np.int16)
        # print(self.polar_reference, np.ceil(check).astype(np.int16))
        if np.any(np.all(self.polar_reference == check, axis=1)):
            phi = check[1]
            theta = check[0]
            key = f"{phi}:{theta}"
            return HInfo(
                hrirs=self.hrirs[key][:],
                hffts=self.hffts[key],
                itds=self.itds[key], 
                coords=check[::-1],
                target=point,
                shape=self.hrir_shape,
                n_neighs=1
            )
        
        cart = point.get_cartesian(mode=self.mode)
        _, indices = self.tree.query([cart.x, cart.y, cart.z], k=neighs)
        
        hinfo = HInfo(
            hrirs=np.empty((neighs, self.hrir_shape[0], self.hrir_shape[1])),
            hffts=[],
            itds=np.empty(neighs, dtype=np.float32),
            coords=np.zeros((neighs, 2)),
            target=point,
            shape=self.hrir_shape,
            n_neighs=neighs
        )
        
        for i, min_index in enumerate(indices):
            coord = self.polar_reference[min_index]
            elev, azim = None, None
            if self.mode == CoordMode.INTERAURAL:
                x, y, z = self.coordinates[min_index, 0], self.coordinates[min_index, 1], self.coordinates[min_index, 2]
                elev_temp = np.arctan2(z, y)
                azim_temp = np.arcsin(x)

                elev = np.rad2deg(elev_temp)
                azim = np.rad2deg(azim_temp) if y >= 0.0 else np.rad2deg(azim_temp) + 180
            else:
                azim, elev = coord
            key = f"{int(elev)}:{int(azim)}"
            hrir = self.hrirs[key][:]
            hinfo.hrirs[i, :, :] = hrir
            hinfo.hffts.append(self.hffts[key])
            hinfo.itds[i] = self.itds[key][()]
            hinfo.coords[i] = np.array([int(elev), int(azim)])
        return hinfo
    
    def build_hrir(self, hrirs: HInfo, method: BuildMode, c: float = 343.3) -> NDArray[np.float64]:
        """
        GENERATE INTERPOLATED DISTANCE-BASED HRIR

        Parameters
        ----------
        hrirs : HInfo
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
        
        if hrirs.n_neighs == 1: 
            return self.__distance_based_hrir(hrir=hrirs.hrirs, rho=hrirs.target.rho, c=c)
        
        interpolated = self.__interpolated_hrir(hrirs=hrirs, method=method, mode=self.interp_domain)
        return self.__distance_based_hrir(hrir=interpolated, rho=hrirs.target.rho, c=c)
    
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
        
        plt.suptitle(f'{title}' "\n" rf'$T = {temp} \degree C$, $RH = {hum} \%$, $P = {pres} Pa$' "\n" rf'$(\rho, \phi, \theta) = {rho, phi, theta}$, $d_0 = {self.source_distance}m$, $d = {rho}m$, $\alpha = {np.round(self.__att_factor, decimals=3)}$, $\gamma = {self.GAMMA}$, $\eta = {self.ETA}$')
        
        ax0.plot(f, ps[:, 0], c="k", lw=0.5)
        ax0.set_title("LEFT CH. POWER SPECTRUM")
        ax0.set_xlabel("Freq.")
        ax0.set_ylabel("Mag.")
        ax1.plot(f, ps[:, 1], c="k", lw=0.5)
        ax1.set_title("RIGHT CH. POWER SPECTRUM")
        ax1.set_xlabel("Freq.")
        ax1.set_ylabel("Mag.")
       
        ax2.plot(self.__frequencies, db_attenuation, c="k", lw=0.5)
        ax2.set_title(f"IMPOSED ISO 9613-1")
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
    