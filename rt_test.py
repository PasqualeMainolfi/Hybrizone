# import section
from hybri import Hybrizone, HybriParams, PolarPoint, AirData, AngleMode, CoordMode, BuildMode, CurveMode, InterpolationDomain
import librosa as lb
import numpy as np
import pyaudio as pa

# main scripts
HRIR_PATH = "/Users/pm/AcaHub/Coding/BinauralSpatial/data/HRIR-KEMAR_DATASET.h5"
RIR_PATH = "/Users/pm/AcaHub/Coding/BinauralSpatial/data/RIR-MIT_SURVEY.h5"
AUDIO_PATH = "/Users/pm/AcaHub/AudioSamples/suzanne_mono.wav"

CHUNK = 1024
SR = 44100
CHANNELS = 2

PARAMS = HybriParams(
        hrir_database=HRIR_PATH, 
        rir_database=RIR_PATH, 
        coord_mode=CoordMode.REGULAR, 
        interp_domain=InterpolationDomain.FREQUENCY,
        build_mode=BuildMode.LINEAR,
        chunk_size=CHUNK,
        interpolation_neighs=3
)

SIG, _ = lb.load(AUDIO_PATH, sr=SR)
SIG_LIMIT = len(SIG)


# main function
def main() -> None:
    # start hybri
    AURALIZER = Hybrizone(params=PARAMS)

    # set air conditions
    air_conditions = AirData(temperature=20, humidity=0.5, pressure=101325.0)
    AURALIZER.imposed_air_conditions(air_data=air_conditions)
    
    # pass the rirs
    AURALIZER.set_rirs(rir1=10, rir2=70, smooth_factor=0.1)
    
    # start parallel process for space data calculation
    AURALIZER.start_space_data_process()

    # audio engine
    PORTAUDIO = pa.PyAudio()
    stream = PORTAUDIO.open(format=pa.paFloat32, channels=CHANNELS, rate=SR, output=True, frames_per_buffer=CHUNK)
    stream.start_stream()

    run = True
    mark = 0
    while True:
        
        try:
            if mark >= SIG_LIMIT:

                # close hybri
                AURALIZER.close()
                run = False
                break
            
            end = min(mark + CHUNK, SIG_LIMIT - 1)
            frame = SIG[mark:end]

            # pass current position
            pos = PolarPoint(rho=1.7, phi=20, theta=60, opt=AngleMode.DEGREE)
            AURALIZER.set_position(position=pos)

            # pass current hybrid space params
            AURALIZER.set_morph_data(direction=0.5, morph_curve=CurveMode.SIGMOID)

            # generates kernels (HRIR and RIR)
            kernels = AURALIZER.get_rir_and_hrir()
            if kernels is not None:

                # auralization
                convolved_frame = AURALIZER.process_frame(frame=frame, kernels=kernels)
                stream.write(convolved_frame.tobytes())

                mark += CHUNK
            else:
                stream.write(np.zeros((CHUNK, 2), dtype=np.float32).tobytes())
        except KeyboardInterrupt:
            run = False
            AURALIZER.close()
            print("[INFO] Process blocked by user!")
            break
            
    while run:
        pass

    stream.stop_stream()
    stream.close()
    PORTAUDIO.terminate()
        

# [MAIN PROGRAM]: if the module is being run as the main program, it calls the "main()" function
if __name__ == "__main__":
    main()