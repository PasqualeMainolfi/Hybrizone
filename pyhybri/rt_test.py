# import section
from hybri import Hybrizone, HybriParams, PolarPoint, AirData, AngleMode, CoordMode, BuildMode, CurveMode, InterpolationDomain, CartesianPoint, LinearTrajectory, ParametricTrajectory
import numpy as np
import pyaudio as pa
from enum import Enum
import os
import wave


# main scripts
HRIR_PATH = "/Users/pm/AcaHub/Coding/BinauralSpatial/data/HRIR-KEMAR_DATASET.h5"
RIR_PATH = "/Users/pm/AcaHub/Coding/BinauralSpatial/data/RIR-MIT_SURVEY.h5"
AUDIO_PATH = "/Users/pm/AcaHub/Coding/BinauralSpatial/audio_examples"

class AudioExample(Enum):
    HELICOPTER = 1,
    GUN_FIRE = 2,
    RUNNING = 3
    
def load_audio_example(audio: AudioExample):
    ex = ""
    match audio:
        case AudioExample.HELICOPTER:
           ex = "helicopter.wav"
        case AudioExample.GUN_FIRE:
           ex = "gun_fire.wav"
        case AudioExample.RUNNING:
           ex = "running.wav"
        case _:
            print("[ERROR] Wrong example")
            exit(1)
        
    path_to_example = os.path.join(AUDIO_PATH, ex)
    s = wave.open(path_to_example, "r")
    return s
    
CHUNK = 1024
SR = 44100
CHANNELS = 2

PARAMS = HybriParams(
        hrir_database_path=HRIR_PATH,
        rir_database_path=RIR_PATH,
        coord_mode=CoordMode.REGULAR,
        interp_domain=InterpolationDomain.FREQUENCY,
        build_mode=BuildMode.SLERPL,
        chunk_size=CHUNK,
        interpolation_neighs=2,
        sample_rate=SR,
        gamma=1.0
)

# main function
def main() -> None:
    AUDIO_SIGNAL = load_audio_example(audio=AudioExample.RUNNING)
    frame_block = AUDIO_SIGNAL.readframes(CHUNK)

    # start hybri
    AURALIZER = Hybrizone(params=PARAMS)

    # set air conditions
    air_conditions = AirData(temperature=20, humidity=0.5, pressure=101325.0)
    AURALIZER.imposed_air_conditions(air_data=air_conditions)

    # pass the rirs
    AURALIZER.set_rirs(rir1=10, rir2=70, smooth_factor=0.1)

    # audio engine
    PORTAUDIO = pa.PyAudio()
    stream = PORTAUDIO.open(format=pa.paFloat32, channels=CHANNELS, rate=SR, output=True, frames_per_buffer=CHUNK)
    stream.start_stream()

    nblocks = AUDIO_SIGNAL.getnframes() // CHUNK
    max_distance = 100
    
        
    # IN-HEAD
    # distances = np.abs(np.linspace(1, -1, nblocks, endpoint=True)) * max_distance
    # azimuths = np.zeros(nblocks)
    # elevations = np.zeros(nblocks)
    
    # FIXED_AZI = 0.0 # IN-FRONT
    # FIXED_PHI = 40 # IN-FRONT
    
    # azimuths[:nblocks // 2 + 1] = FIXED_AZI 
    # azimuths[nblocks // 2 + 1:] = FIXED_AZI + 180 # invert azi to back FLY-BY
    # elevations[:nblocks // 2 + 1] = FIXED_PHI
    # elevations[nblocks // 2 + 1:] = -FIXED_PHI # invert ele to back
    
    
    hel_cpa = np.array([0.0, -3.0, 0.0]) # closest point to me 
    # hel_cpa = np.array([0.0, 0.0, 5.0]) # closest point to me fli-by
    hel_direction = np.array([-1.0, 0.0, 0.0])
    hel_distances = np.linspace(-max_distance, max_distance, nblocks, endpoint=True)
    hel_linear_traj = LinearTrajectory(cpa=hel_cpa, direction=hel_direction)
    hel_points = hel_linear_traj.get_points(distances=hel_distances)
    
    omega = np.linspace(0.1, 1.5, nblocks, endpoint=True)
    times = np.linspace(0, nblocks * CHUNK / SR, nblocks, endpoint=True)
    distances_circular = np.linspace(10, 1, nblocks, endpoint=True)
    vertical_vel = np.zeros(nblocks)
    # vertical_vel[:nblocks // 2 + 1] = 0.5
    # vertical_vel[nblocks // 2 + 1:] = -0.5
    hel_circular_path = ParametricTrajectory.get_circular_points(omega=omega, t=times, radius=distances_circular, start_elevation=0.0, vertical_vel=vertical_vel)
    parabolic = ParametricTrajectory.get_parabolic_points(total_distance=max_distance, total_duration=nblocks * CHUNK / SR, max_elevation=30, t=times)

    fixed_point = PolarPoint(rho=10, phi=25, theta=270, opt=AngleMode.DEGREE)
    
    
    mode = "fixed"
    
    RUN = True
    index = 0
    while frame_block != b'':

        try:
            if index < nblocks:
            #     current_rho = distances[index]
            #     current_phi = elevations[index]
            #     current_theta = azimuths[index]
                if mode == "linear":
                    pos = hel_points[index]
                elif mode == "circular":
                    pos = hel_circular_path[index]
                elif mode == "parabolic":
                    pos = parabolic[index]
                elif mode == "fixed":
                    pos = fixed_point
            else:
            #     current_rho = distances[-1]
            #     current_phi = elevations[-1]
            #     current_theta = azimuths[-1]
                if mode == "linear":
                    pos = hel_points[-1]
                elif mode == "circular":
                    pos = hel_circular_path[-1]
                elif mode == "parabolic":
                    pos = parabolic[-1]
                elif mode == "fixed":
                    pos = fixed_point
            
            frame = np.frombuffer(frame_block, dtype=np.int16) # mono
            frame = frame / 32768.0
            
            print(f"rho: [{pos.rho}], phi: [{pos.phi}], theta: [{pos.theta}]")
            
            # pos = PolarPoint(rho=current_rho, phi=current_phi, theta=current_theta, opt=AngleMode.DEGREE)
            AURALIZER.set_position(position=pos)

            # pass current hybrid space params
            AURALIZER.set_morph_data(direction=0.37, morph_curve=CurveMode.LINEAR)

            # generates kernels (HRIR and RIR)
            kernels = AURALIZER.get_kernels()

            # auralization
            convolved_frame = AURALIZER.process_frame(frame=frame, kernels=kernels)
            stream.write(convolved_frame.tobytes())

            frame_block = AUDIO_SIGNAL.readframes(CHUNK)
            index += 1
        except KeyboardInterrupt:
            RUN = False
            AURALIZER.close()
            print("[INFO] Process blocked by user!")
            break

    while RUN and frame_block != b'':
        pass
    

    AURALIZER.close()
    stream.stop_stream()
    stream.close()
    PORTAUDIO.terminate()
    AUDIO_SIGNAL.close()

# [MAIN PROGRAM]: if the module is being run as the main program, it calls the "main()" function
if __name__ == "__main__":
    main()
