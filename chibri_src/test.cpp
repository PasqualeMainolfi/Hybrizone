#include "src/htools.hpp"
#include "src/hybri.hpp"
#include <cstddef>
#include <cstdlib>
#include <portaudio.h>
#include <sndfile.hh>
#include <vector>
#include <chrono>

#define HRIR_DATASET_PATH ("/Users/pm/AcaHub/Coding/BinauralSpatial/data/HRIR-KEMAR_DATASET.h5")
#define RIR_DATASET_PATH ("/Users/pm/AcaHub/Coding/BinauralSpatial/data/RIR-MIT_SURVEY.h5")
#define AUDIO_FILE ("/Users/pm/AcaHub/AudioSamples/suzanne_mono.wav")
#define FS (44100.0)
#define CHUNK (2048)

#define CHECK_PORTAUDIO(err) \
    do { \
        if ((err) != paNoError) { \
            std::cerr << "[ERROR] Portaudio error: " << Pa_GetErrorText(err) << std::endl; \
            std::exit(1); \
        } \
    } while (0) \

int main(void) {

    SF_INFO sfinfo;
    sfinfo.format = 0;
    SNDFILE* audio_file_in = sf_open(AUDIO_FILE, SFM_READ, &sfinfo);

    PaError perr;
    const PaDeviceInfo* info;
    PaStreamParameters output_params;

    perr = Pa_Initialize();
    CHECK_PORTAUDIO(perr);

    output_params.device = Pa_GetDefaultOutputDevice();
    info = Pa_GetDeviceInfo(output_params.device);

    output_params.channelCount = 2;
    output_params.sampleFormat = paFloat32;
    output_params.suggestedLatency = info->defaultHighOutputLatency;
    output_params.hostApiSpecificStreamInfo = NULL;

    PaStream* stream;
    perr = Pa_OpenStream(&stream, nullptr, &output_params, (double) FS, CHUNK, paClipOff, nullptr, nullptr);
    CHECK_PORTAUDIO(perr);
    perr = Pa_StartStream(stream);
    CHECK_PORTAUDIO(perr);

    AirData air_data;
    Hybrizone hybri(HRIR_DATASET_PATH, RIR_DATASET_PATH, CHUNK, FS);
    hybri.set_air_condition(air_data);
    hybri.set_rirs(10, 30, 0.1);

    double azi = 0.0;
    double rho = 0.0;
    double rho_step = 1.0;
    bool rho_flag = false;
    double max_distance = 30.0;
    while (true) {

        std::vector<double> frame(CHUNK);
        sf_count_t fcount = sf_read_double(audio_file_in, frame.data(), CHUNK);
        if (fcount <= 0) break;

        // process time start
        auto start_time = std::chrono::high_resolution_clock::now();

        PolarPoint target(rho, 0.0, azi, AngleMode::DEGREE);
        hybri.set_target_position(target);

        if (rho > max_distance) rho_flag = true;
        if (rho < rho_step) rho_flag = false;
        rho = rho_flag ? rho - rho_step : rho + rho_step;
        azi = azi >= 360.0 ? 0.0 : azi + 25.0;

        hybri.set_hybrid_rir_params(0.37, CurveMode::SIGMOID);

        Kernels kernels;
        hybri.generate_kernels(&kernels);

        // generate kernels time
        auto time_to_generate_kernels = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_kernels = time_to_generate_kernels - start_time;

        HybriOuts channels;
        hybri.process_frame(&channels, frame.data(), &kernels);

        // time to convolve
        auto time_to_convolve = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_convolve = time_to_convolve - start_time;

        std::vector<float> interleaved = channels.get_float_interleaved();
        Pa_WriteStream(stream, interleaved.data(), CHUNK);

        // total time
        auto total_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_time = total_time - start_time;

    #ifdef DEBUG
        // get times
        std::cerr << "[INFO PROCESS TIME]" << std::endl;
        std::cerr << "Time to generate kernels: " << t_kernels.count() <<  std::endl;
        std::cerr << "Time to convolve frame and kernels: " << t_convolve.count() << std::endl;
        std::cerr << "Total time: " << t_time.count() << std::endl;
    #endif
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    return 0;
}
