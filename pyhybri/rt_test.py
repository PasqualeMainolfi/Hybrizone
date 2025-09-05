# import section
from hybri import (
    Hybrizone,
    HybriParams,
    PolarPoint, # noqa: F401
    AirData,
    AngleMode, # noqa: F401
    CoordMode,
    BuildMode,
    CurveMode,
    InterpolationDomain,
    CartesianPoint, # noqa: F401
    LinearTrajectory,
    ParametricTrajectory,
    linear_direction_info
)
import numpy as np
import pyaudio as pa
from enum import Enum
import os
import wave
import pygame


# main scripts
HRIR_PATH = "/Users/pm/AcaHub/Coding/BinauralSpatial/data/HRIR-KEMAR_DATASET.h5"
RIR_PATH = "/Users/pm/AcaHub/Coding/BinauralSpatial/data/RIR-MIT_SURVEY.h5"
AUDIO_PATH = "/Users/pm/AcaHub/Coding/BinauralSpatial/audio_examples"

class AudioExample(Enum):
    HELICOPTER1 = 1,
    GUN_FIRE = 2,
    RUNNING = 3
    HELICOPTER2 = 4,
    FOOTSTEP = 5,
    JET = 6,
    WALKING_GRASS = 7,
    SOVIET_GUNFIRE = 8,
    GUNSHOT = 9,
    STAIRS = 10


def load_audio_example(audio: AudioExample):
    ex = ""
    match audio:
        case AudioExample.HELICOPTER1:
           ex = "helicopter1.wav"
        case AudioExample.HELICOPTER2:
           ex = "helicopter2.wav"
        case AudioExample.FOOTSTEP:
           ex = "footstep.wav"
        case AudioExample.JET:
           ex = "jet.wav"
        case AudioExample.GUN_FIRE:
           ex = "gun_fire.wav"
        case AudioExample.RUNNING:
           ex = "running.wav"
        case AudioExample.WALKING_GRASS:
           ex = "walking-grass.wav"
        case AudioExample.SOVIET_GUNFIRE:
           ex = "soviet-gunfire.wav"
        case AudioExample.GUNSHOT:
           ex = "gunshot.wav"
        case AudioExample.STAIRS:
           ex = "stairs.wav"
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
        gamma=1.2
)

pygame.init()

# main function
def main() -> None:
    SCREEN = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    
    # AUDIO_SIGNAL = load_audio_example(audio=AudioExample.FOOTSTEP)
    # frame_block = AUDIO_SIGNAL.readframes(CHUNK)

    # start hybri
    AURALIZER = Hybrizone(params=PARAMS)

    # set air conditions
    air_conditions = AirData(temperature=20, humidity=0.5, pressure=101325.0)
    AURALIZER.imposed_air_conditions(air_data=air_conditions)

    # pass the rirs
    AURALIZER.set_rirs(rir1=36, rir2=39, smooth_factor=0.1)

    # audio engine
    PORTAUDIO = pa.PyAudio()
    stream = PORTAUDIO.open(format=pa.paFloat32, channels=CHANNELS, rate=SR, output=True, frames_per_buffer=CHUNK)
    stream.start_stream()

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

    AUDIO_SIGNAL = load_audio_example(audio=AudioExample.HELICOPTER2)
    frame_block = AUDIO_SIGNAL.readframes(CHUNK)
    nblocks = AUDIO_SIGNAL.getnframes() // CHUNK

    MAX_DISTANCE = 30
    example_mode = "circular_traj"

    points_pol, points_car = None, None
    match example_mode:
        case "linear_traj":
            cpa = np.array([1.0, 0.0, -3.0]) # closest point (left Y pos, in front X pos, up Z pos)
            direction = np.array([0.0, 1.0, -3.0])

            linear_direction_info(direction=direction)

            distances = np.linspace(-MAX_DISTANCE / 2, MAX_DISTANCE / 2, nblocks, endpoint=True)
            
            space_step = MAX_DISTANCE / (nblocks - 1)
            sec = space_step / (CHUNK / SR) 
            print(f"[VELOCITY] = {sec:.3f} m/s")
            
            linear_traj = LinearTrajectory(cpa=cpa, direction=direction)
            points_pol, points_car = linear_traj.get_points(distances=distances)
        case "circular_traj":
            omega = np.linspace(0.1, 10, nblocks, endpoint=True)
            times = np.linspace(0, nblocks * CHUNK / SR, nblocks, endpoint=True)
            distances_circular = np.linspace(10, 1, nblocks, endpoint=True)
            start_elevelation_pos = 10
            vertical_vel = np.zeros(nblocks) - 1
            # vertical_vel[:nblocks // 2 + 1] = 0.5
            # vertical_vel[nblocks // 2 + 1:] = -0.5

            points_pol, points_car = ParametricTrajectory.get_circular_points(omega=omega, t=times, radius=distances_circular, start_elevation=start_elevelation_pos, vertical_vel=vertical_vel)

    SCALE = 50
    GAIN = 1
    RUN = True
    
    print(f"[INFO] Sound speed: {AURALIZER.sound_speed:.5f} m/s")
    
    index = 0
    while frame_block != b'':
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUN = False

        SCREEN.fill("black")

        try:
            if index < nblocks:
            #     current_rho = distances[index]
            #     current_phi = elevations[index]
            #     current_theta = azimuths[index]
                pos = points_pol[index]
                cart = points_car[index]
            else:
            #     current_rho = distances[-1]
            #     current_phi = elevations[-1]
            #     current_theta = azimuths[-1]
                pos = points_pol[-1]
                cart = points_car[-1]

            frame = np.frombuffer(frame_block, dtype=np.int16) # mono
            frame = (frame / 32768.0) * GAIN

            # print(f"rho: [{pos.rho}], phi: [{pos.phi}], theta: [{pos.theta}]")
            
            depth = cart.x + 1e-12
            world_x = cart.y
            world_y = cart.z
            
            x_screen = (-world_x * SCALE / depth) + SCREEN.get_width() / 2
            y_screen = SCREEN.get_height() / 2 - (world_y * SCALE / depth)
            
            radius = 1 * SCALE / depth
            print(f"x: [{x_screen}], y: [{y_screen}], DEPTH: [{radius}]")
            
            pygame.draw.circle(SCREEN, "white", center=(x_screen, y_screen), radius=radius)

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
            pygame.display.flip()
            clock.tick(60)
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
    
    pygame.quit()

# [MAIN PROGRAM]: if the module is being run as the main program, it calls the "main()" function
if __name__ == "__main__":
    main()
