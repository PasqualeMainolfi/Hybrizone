#ifndef RBUILDER_HPP
#define RBUILDER_HPP

#include "htools.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

class RBuilder
{
public:
    RirDatasetRead* dataset;

    RBuilder(const char* dataset_path, double source_distance, double sample_rate) {
        this->dataset = new RirDatasetRead(dataset_path);

        if (this->dataset->get_sample_rate() != sample_rate) {
            std::cerr << "[ERROR] Sample rate must be the same of the RIRs!" << std::endl;
            std::exit(1);
        }

        this->source_distance = source_distance;
        this->fs = sample_rate;
        this->cache = new LRUCache(CACHE_CAPACITY, CacheType::RIR);
        this->current_cache_key = "";
        this->prev_distance = -1.0;
        this->prev_distance_rir = std::vector<double>();

        this->geometric_attenuation = new GeometricAttenuation(this->fs, 1);
        this->db_attenuation = std::vector<double>();
        this->iso9613 = nullptr;

        this->cross_fade_length = static_cast<size_t>(INTERNAL_KERNEL_TRANSITION * this->fs);
        this->cross_fade_coeff_a = std::vector<double>(this->cross_fade_length);
        this->cross_fade_coeff_b = std::vector<double>(this->cross_fade_length);
        for (size_t i = 0; i < this->cross_fade_length; ++i) {
            double alpha_linear =  static_cast<double>(i) / static_cast<double>(this->cross_fade_length - 1);
            double alpha = alpha_linear * M_PI_2;
            this->cross_fade_coeff_a[i] = std::cos(alpha);
            this->cross_fade_coeff_b[i] = std::sin(alpha);
        }
    }

    ~RBuilder() {
        delete this->dataset;
        delete this->cache;
        delete this->iso9613;
        delete this->geometric_attenuation;
    }

    void set_air_condition(AirData* air_data) {
        this->iso9613 = new ISO9613Filter(air_data, this->fs);
        this->db_attenuation.resize(NFREQS);
        this->iso9613->get_attenuation_air_absorption(this->db_attenuation.data());
    }

    void set_rirs(size_t index_a, size_t index_b, double smooth_factor) {
        this->current_cache_key = std::format("{}:{}:{}", index_a, index_b, smooth_factor);

        if (!this->cache->contains(this->current_cache_key)) {
            RirFromDataset rir_a;
            this->dataset->get_rdata(&rir_a, index_a);
            RirFromDataset rir_b;
            this->dataset->get_rdata(&rir_b, index_b);

            size_t rir_size = std::max(rir_a.lenght, rir_b.lenght);
            // std::cout << rir_size << " : " << rir_a.lenght << " : " << rir_b.lenght << std::endl;

            if (rir_a.lenght < rir_b.lenght) {
                std::vector<double> temp(rir_b.lenght, 0.0);
                memcpy(temp.data(), rir_a.rir.data(), sizeof(double) * rir_a.lenght);
                rir_a.rir = temp;
                rir_a.lenght = rir_b.lenght;
            }

            if (rir_a.lenght > rir_b.lenght) {
                std::vector<double> temp(rir_a.lenght, 0.0);
                memcpy(temp.data(), rir_b.rir.data(), sizeof(double) * rir_b.lenght);
                rir_b.rir = temp;
                rir_b.lenght = rir_a.lenght;
            }

            size_t fftw_size = rir_size / 2 + 1;

            Morphdata md;
            md.source_a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftw_size);
            md.source_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftw_size);
            md.morphed = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftw_size);
            md.smooth_factor = smooth_factor;
            md.lenght = rir_size;
            md.fftw_length = fftw_size;

            this->get_fft(rir_a.rir.data(), md.source_a, rir_size);
            this->get_fft(rir_b.rir.data(), md.source_b, rir_size);

            std::vector<double> scep(fftw_size, 0.0);
            this->get_spectral_envelope(md.source_a, scep.data(), rir_size, fftw_size, smooth_factor);

            std::vector<double> tcep(fftw_size, 0.0);
            this->get_spectral_envelope(md.source_b, tcep.data(), rir_size, fftw_size, smooth_factor);

            for (size_t i = 0; i < fftw_size; ++i) {
                std::complex<double> target_flatten(md.source_b[i][0], md.source_b[i][1]);
                target_flatten = (target_flatten / (tcep[i] + EPSILON)) * scep[i];
                md.morphed[i][0] = target_flatten.real();
                md.morphed[i][1] = target_flatten.imag();
            }

            this->cache->put(this->current_cache_key, (Morphdata*) new Morphdata(md));
        }
    }

    void build_hybrid_rir(Rir* rir, double direction, CurveMode curve_mode, double distance) {
        if (this->current_cache_key == "") {
            std::cerr << "[ERROR] Set RIRs first!" << std::endl;
            std::exit(1);
        }

        Morphdata* md = (Morphdata*) this->cache->get(this->current_cache_key);

        double factor = this->non_linear_morph_curve(direction, curve_mode);
        double sx = std::max(1.0 - 2.0 * factor, 0.0);
        double cx = 1.0 - std::abs(1.0 - 2.0 * factor);
        double dx = std::max(2.0 * factor - 1.0, 0.0);

        fftw_complex* y_spectrum = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * md->fftw_length);
        for (size_t i = 0; i < md->fftw_length; ++i) {
            y_spectrum[i][0] = sx * md->source_a[i][0] + cx * md->morphed[i][0] + dx * md->source_b[i][0];
            y_spectrum[i][1] = sx * md->source_a[i][1] + cx * md->morphed[i][1] + dx * md->source_b[i][1];
        }

        md->execute_ifft(y_spectrum);
        std::vector<double> temp_morphed = md->out_morphed;
        fftw_free(y_spectrum);

        double max_morphed = *std::max_element(
            temp_morphed.begin(), temp_morphed.end(), [](double a, double b) {
                return std::abs(a) < std::abs(b);
            }
        );

        for (auto& val : temp_morphed) val /= max_morphed;

        this->apply_distance(temp_morphed.data(), distance, md->lenght);
        std::vector<double> current_temp = temp_morphed;

        if (this->prev_distance != -1.0 && this->prev_distance != distance) {
            double d = std::abs(distance - this->prev_distance);
            if (d > MAX_CROSSFADE_DISTANCE) {
                cross_fade(this->prev_distance_rir.data(), current_temp.data(), this->cross_fade_coeff_a.data(), this->cross_fade_coeff_b.data(), this->cross_fade_length);
            }
        }

        this->prev_distance = distance;
        this->prev_distance_rir = temp_morphed;

        rir->rir.resize(md->lenght);
        memcpy(rir->rir.data(), current_temp.data(), sizeof(double) * md->lenght);
        rir->length = md->lenght;
    }

private:
    LRUCache* cache;
    std::string current_cache_key;
    double prev_distance;
    std::vector<double> prev_distance_rir;
    double fs;
    ISO9613Filter* iso9613;
    std::vector<double> db_attenuation;
    GeometricAttenuation* geometric_attenuation;
    double source_distance;
    size_t cross_fade_length;
    std::vector<double> cross_fade_coeff_a;
    std::vector<double> cross_fade_coeff_b;

    void get_spectral_envelope(fftw_complex* x, double* y, size_t frame_size, size_t fft_size, double smooth_factor) {
        fftw_complex* db = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
        double mag_max = std::abs(std::complex<double>(x[0][0], x[0][1]));
        for (size_t i = 0; i < fft_size; ++i) {
            double mag = std::hypot(x[i][0], x[i][1]);
            mag_max = std::max(mag_max, mag);
            db[i][0] = std::log10(mag + EPSILON);
            db[i][1] = 0.0;
        }

        std::vector<double> rc(frame_size, 0.0);
        fftw_plan ifft = fftw_plan_dft_c2r_1d(frame_size, db, rc.data(), FFTW_ESTIMATE);
        fftw_execute(ifft);
        fftw_destroy_plan(ifft);

        for (auto& val : rc) val /= static_cast<double>(frame_size);

        fftw_complex* realcep = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
        fftw_plan fft = fftw_plan_dft_r2c_1d(frame_size, rc.data(), realcep, FFTW_ESTIMATE);
        fftw_execute(fft);
        fftw_destroy_plan(fft);

        std::vector<double> realcep_real(fft_size, 1.0);
        for (size_t i = 0; i < fft_size; ++i) realcep_real[i] = realcep[i][0];
        fftw_free(realcep);

        double rc_mean = std::accumulate(realcep_real.begin(), realcep_real.end(), 0.0) / static_cast<double>(fft_size);
        for (auto& val : realcep_real) val = std::exp(val - rc_mean);

        size_t kernel_length = static_cast<size_t>(fft_size * smooth_factor);
        std::vector<double> smooth_kernel(kernel_length, 1.0);
        for (size_t i = 0; i < kernel_length; ++i) smooth_kernel[i] = static_cast<double>(i) / static_cast<double>(kernel_length);

        std::vector<double> rc_smoothed;
        fft_convolve(&rc_smoothed, realcep_real.data(), smooth_kernel.data(), fft_size, kernel_length, ConvMode::SAME);

        double max_smooth = *std::max_element(
            rc_smoothed.begin(), rc_smoothed.end(), [](double a, double b) {
                return std::abs(a) < std::abs(b);
            }
        );

        double scale_factor = mag_max / (max_smooth + EPSILON);
        for (size_t i = 0; i < fft_size; ++i) y[i] = rc_smoothed[i] * scale_factor;
    }

    double non_linear_morph_curve(double direction, CurveMode curve_mode) {
        double value = 0.0;
        switch (curve_mode) {
            case CurveMode::LINEAR:
                value = direction;
                break;
            case CurveMode::SIGMOID:
                value = 1.0 / (1.0 + std::exp(-10.0 * (direction - 0.5)));
                break;
            case CurveMode::EXPONENTIAL:
                value = direction * direction;
                break;
            case CurveMode::LOGARITHMIC:
                value = std::log10(direction * 9 + 1);
                break;
        }
        return value;
    }

    void apply_distance(double* x, double distance, size_t frame_length) {
        double factor = this->geometric_attenuation->calculate_geometric_factor(this->source_distance, distance);
        factor = factor > 1.0 ? 1.0 : factor;
        double d = distance - std::max(this->source_distance, ETA);
        this->iso9613->air_absorption_filter(x, this->db_attenuation.data(), d, frame_length);
        for (size_t i = 0; i < frame_length; ++i) x[i] *= factor;
    }

    void get_fft(double* x, fftw_complex* y, size_t length) {
        fftw_plan p = fftw_plan_dft_r2c_1d(length, x, y, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
    }
};

#endif
