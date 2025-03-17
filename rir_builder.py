from common_modules import np, h5py, NDArray, Enum, plt, gridspec
import hashlib

class CurveMode(Enum):
    LINEAR = 0
    SIGMOID = 1
    LOGARITHMIC = 2
    EXPONENTIAL = 3
    
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

class RIRMorpha():
    def __init__(self, rir_database: str):
        """
        Parameters
        ----------
        rir_database : str
            path to h5 database
        """
        
        self.rirs = h5py.File(rir_database, "r")
        self.fs = self.rirs.attrs["fs"]
        self.source1 = None
        self.source2 = None
        self.morphed = None
        self.length = None
        self.__cache = {"r1": None, "r2": None, "sf": None}
    
    def close(self) -> None:
        self.rirs.close()
    
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
        
        k1 = self.rirs.attrs["IR-keys"][rir1]
        k2 = self.rirs.attrs["IR-keys"][rir2]
        source1 = self.rirs[k1][:]
        source2 = self.rirs[k2][:]
        n1 = len(source1)
        n2 = len(source2)
        if n1 < n2: source1 = np.pad(source1, (0, n2 - n1), constant_values=0.0, mode="constant")
        elif n1 > n2: source2 = np.pad(source2, (0, n1 - n2), constant_values=0.0, mode="constant")
        self.length = max(n1, n2)
        
        scep = self.__get_spectral_envelope(source=source1, smooth_factor=smooth_factor)
        tcep = self.__get_spectral_envelope(source=source2, smooth_factor=smooth_factor)
        self.source1 = np.fft.rfft(source1)
        self.source2 = np.fft.rfft(source2)
        target_flatten = self.source2 / (tcep + 1e-12)
        self.morphed = scep * target_flatten
        self.__cache["r1"] = rir1
        self.__cache["r2"] = rir2
        self.__cache["sf"] = smooth_factor
    
    def set_rir1(self, rir1: int) -> None:
        self.set_rirs(rir1=rir1, rir2=self.__cache["r2"], smooth_factor=self.__cache["sf"])
    
    def set_rir2(self, rir2: int) -> None:
        self.set_rirs(rir1=self.__cache["r1"], rir2=rir2, smooth_factor=self.__cache["sf"])
    
    def set_smooth_factor(self, smooth_factor: float) -> None:
        self.set_rirs(rir1=self.__cache["r1"], rir2=self.__cache["r2"], smooth_factor=smooth_factor)
        
    def __get_spectral_envelope(self, source: NDArray[np.float32], smooth_factor: float) -> NDArray[np.float32]:
        fft_source = np.fft.rfft(source)
        mag = np.abs(fft_source)
        log = np.log10(mag)
        realcp = np.fft.irfft(log + 1e-12).real
        realcp = np.fft.rfft(realcp).real
        realcp_mean = np.mean(realcp)
        realcp = np.exp(realcp - realcp_mean)
        
        kernel_length = int(len(realcp) * smooth_factor)
        kernel = np.ones(kernel_length) / smooth_factor
        rc_smoothed = np.convolve(realcp, kernel, mode="same")
        
        scale_factor = np.max(mag) / (np.max(rc_smoothed) + 1e-12)
        return rc_smoothed * scale_factor
    
    def __nonlinear_morphing_curve(self, direction: float, curve_type: CurveMode) -> float:
        match curve_type:
            case CurveMode.LINEAR: return direction
            case CurveMode.SIGMOID: return 1 / (1 + np.exp(-10 * (direction - 0.5)))
            case CurveMode.EXPONENTIAL: return direction**2
            case CurveMode.LOGARITHMIC: return np.log10(direction * 9 + 1)
            case _:
                print("[ERROR] Curve mode not implemented!")
                exit(1)
    
    def morpha(self, direction: float, morph_curve: CurveMode) -> NDArray[np.float32]:
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
        
        if self.source1 is None or self.source2 is None or self.morphed is None:
            print("[ERROR] Set RIRs first!")
            exit(1)
            
        mf = self.__nonlinear_morphing_curve(direction=direction, curve_type=morph_curve)
        
        sx = max(1.0 - 2.0 * mf, 0.0)  
        cx = 1.0 - abs(1.0 - 2.0 * mf)
        dx = max(2.0 * mf - 1.0, 0.0)
        
        yspectrum = (sx * self.source1 + cx * self.morphed + dx * self.source2) * 0.5
        y = np.fft.irfft(yspectrum)
        y /= np.max(np.abs(y) + 1e-12)
        return y
    
    def __get_data_plot(self, rir: NDArray[np.float32]) -> PlotData:
        n = len(rir)
        e = np.cumsum(rir[::-1] ** 2)[::-1]
        e /= np.max(e)
        db = 10 * np.log10(e + 1e-12)
        mag = np.abs(np.fft.rfft(rir))
        t = np.arange(n) / self.fs
        freqs = np.fft.rfftfreq(len(t), d=1 / self.fs)
        return PlotData(t=t, f=freqs, mag=mag, integr=db)
    
    def plot_rir(self, rir: int|NDArray[np.float32], title: str|None = None) -> None:
        """
        PLOT RIR

        Parameters
        ----------
        rir : int | NDArray[np.float32]
            rir index or rir data array
        
        title : str | None
            plot title, by default None
        """
        
        title_ = None
        if isinstance(rir, int):
            rir_key = self.rirs.attrs["IR-keys"][rir]
            rir = self.rirs[rir_key][:]
            title_ = rir_key

        plot_data = self.__get_data_plot(rir=rir)
        
        _ = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        if title is None:
            if title_ is None:
                title_ = "No Title"
        else:
            title_ = title
        
        plt.suptitle(f"RIR: {title_}")
        
        ax0 = plt.subplot(gs[0, :])
        ax1 = plt.subplot(gs[1, 0])
        ax2 = plt.subplot(gs[1, 1])
        
        ax0.plot(plot_data.t, rir, c="k", lw=0.7)
        ax0.set_xlabel("Time")
        ax0.set_ylabel("Amplitude")
    
        ax1.plot(plot_data.f, plot_data.mag, c="k", lw=0.7)
        ax1.set_xlabel("Freqs")
        ax1.set_ylabel("Mag")
        ax1.set_title("Power Spectrum")
        
        ax2.plot(plot_data.t, plot_data.integr, c="k", lw=0.7)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("dB")
        ax2.set_title("Inverse Integration (Energy Decay Curve)")
        
        plt.tight_layout()
        plt.show()