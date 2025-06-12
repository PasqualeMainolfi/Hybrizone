# HYBRIZONE

## Real-Time Distance-Extended Binaural Auralization in Hybrid Acoustic Spaces

This repository contains the development version of a framework for real-time immersive binaural audio rendering.  

The model unifies direct sound and environmental reflections using only data from real-world acoustic measurements. It extends standard HRIRs by introducing distance as a parameter and integrates a spectral hybridization method for Room Impulse Responses (RIRs). An adaptive algorithm refines spatial perception through fractional delays, geometric attenuation, and dynamic filtering, simulating air absorption according to ISO 9613-1:1993.  

The system also supports nonlinear spectral morphing of multiple RIRs to create hybrid acoustic spaces that, while physically plausible, may not occur naturally. Real-time signal continuity is ensured through dynamic convolutional crossfades that prevent artifacts during kernel transitions while maintaining efficiency.  

### Dataset requirements for using the framework

To use the framework, you first need to generate two compatible .h5 datasets with a specific structure. The framework expects .h5 files formatted as follows:  

One for HRIRs (Head-Related Impulse Responses):

1. Dataset name `polar_index` representing polar coordinates (azimuth \theta, elevation \phi) with shape (n, 2).
2. Dataset named `interaural_coords` of Cartesian coordinates in interaural mode, defined as: ($x = \sin(\theta)$, $y = \cos(\theta)\cos(\phi)$, $z = \cos(\theta)\sin(\phi)$) with shape (n, 3).
3. Dataset named `regular_coordinates`: Integer indices of Cartesian coordinates in regular mode, defined as:
($x = \sin(\theta)\cos(\phi)$, $y = \cos(\theta)\cos(\phi)$, $z = \sin(\phi)$)
with shape (n, 3).
1. Group `hrir`: Contains sub-datasets for each coordinate set (named `elev_azim`), each with HRIRs in the time domain, shape (n, 2), with no ITD (Interaural Time Difference).
2. Group `hrir_fft`: For each coordinate subgroup named `elev_azim`, contains datasets `fft`, `mag`, and `angle`, representing the frequency domain HRIR components without ITD.
3. Group `hrir_itd`: Contains datasets for each coordinate `elev_azim` set with the extracted ITD values in seconds using the provided scripts `get_itd.py`.
4. Attribute `hrir_shape`: Specifies the shape of the HRIR data.
5. Attribute `fs`: Sampling frequency (float).
6. Attribute `source_distance`: Distance of the sound source during recording (float).

One for RIRs (Room Impulse Responses)

1. Attribute `IR-keys`: A 1D array of strings containing the names of all available RIR entries.
2. Attribute `fs`: Sampling rate used during recording, as a float (e.g., 48000.0).
3. Attribute `encoding`: String indicating the audio encoding format (e.g., "pcm16", "float32").
4. Dataset `<rir_key>`: For each RIR name listed in **IR-keys** attribute, there must be a corresponding dataset containing the mono RIR signal. Each RIR represents the room’s impulse response to a specific source-microphone configuration.

### Using framework

To run the Python prototype, see `/pyhybri/rt_test.py` to use the python prototype.
If you prefer to use the C++ version, navigate to the `/chybri` directory and compile it with:

```shell
make build
make run
```

You can use `/chybri/test.cpp` as a reference for how to use the C++ API.  

The C++ version is operational, but still in development and subject to change.  
To measure and print processing time, use the `make debug`