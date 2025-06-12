#ifndef HYBRI_HPP
#define HYBRI_HPP

#include "hbuilder.hpp"
#include "htools.hpp"
#include "rbuilder.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>


struct Kernels
{
    Hrir* hrir;
    Rir* rir;

    Kernels()
    : hrir(nullptr), rir(nullptr)
    { }

    Kernels(const Hrir& h, const Rir& r)
    {
        this->hrir = new Hrir(h);
        this->rir = new Rir(r);
    }

    ~Kernels() {
        if (this->hrir) delete this->hrir;
        if (this->rir) free(this->rir);
    }
};

class Hybrizone
{
public:
    HBuilder* hbuilder;
    RBuilder* rbuilder;

    Hybrizone(const char* hrir_dataset_path, const char* rir_dataset_path, size_t chunk_size, double sample_rate)
    : chunk(chunk_size), fs(sample_rate)
    {
        this->hbuilder = new HBuilder(hrir_dataset_path);
        this->hkernel = new Hrir();
        this->hleft_buffer = new OSAConv(chunk_size);
        this->hright_buffer = new OSAConv(chunk_size);

        if (rir_dataset_path) {
            this->rbuilder = new RBuilder(rir_dataset_path, this->hbuilder->dataset->get_source_distance(), sample_rate);
            this->rkernel = new Rir();
            this->rmono_buffer = new OSAConv(chunk_size);
        } else {
            this->rbuilder = nullptr;
            this->rkernel = nullptr;
            this->rmono_buffer = nullptr;
        }

        this->morph_direction = 0.0;
        this->morph_curve = CurveMode::LINEAR;

        this->distance = 0.0;

        if (this->fs != this->hbuilder->fs) {
            std::cerr << "[ERROR] Sample rate must be equal to HRIR's sample rate!" << std::endl;
            std::exit(1);
        }

        this->mono = std::vector<double>(this->chunk, 0.0);
    }

    ~Hybrizone() {
        delete this->hbuilder;
        delete this->hkernel;
        delete this->hleft_buffer;
        delete this->hright_buffer;

        if (this->rbuilder) {
            delete this->rbuilder;
            delete this->rkernel;
            delete this->rmono_buffer;
        }
    }

    void set_air_condition(AirData air_data) {
        this->hbuilder->set_air_condition(&air_data);
        if (this->rbuilder) this->rbuilder->set_air_condition(&air_data);
    }

    void set_rirs(size_t index_a, size_t index_b, double smooth_factor) {
        this->rbuilder->set_rirs(index_a, index_b, smooth_factor);
    }

    void set_target_position(PolarPoint p) {
        this->hbuilder->hmatching(&p);
        this->distance = p.rho;
    }

    void set_hybrid_rir_params(double direction, CurveMode curve_mode) {
        this->morph_direction = direction;
        this->morph_curve = curve_mode;
    }

    void generate_kernels(Kernels* kernels) {
        this->hbuilder->build_kernel(this->hkernel);
        kernels->hrir = new Hrir(*this->hkernel);
        if (this->rbuilder != nullptr) {
            this->rbuilder->build_hybrid_rir(this->rkernel, this->morph_direction, this->morph_curve, this->distance);
            kernels->rir = new Rir(*this->rkernel);
        }
    }

    void process_frame(HybriOuts* out_channels, double* frame, Kernels* kernels) {
        if (this->rbuilder != nullptr) {
            this->rmono_buffer->process(this->mono.data(), frame, kernels->rir->rir, kernels->rir->length);
        } else {
            memcpy(this->mono.data(), frame, sizeof(double) * this->chunk);
        }

        out_channels->left_channel.resize(this->chunk);
        out_channels->right_channel.resize(this->chunk);
        out_channels->buffer_size = this->chunk;

        this->hleft_buffer->process(out_channels->left_channel.data(), this->mono.data(), kernels->hrir->left_channel, kernels->hrir->channel_length);
        this->hright_buffer->process(out_channels->right_channel.data(), this->mono.data(), kernels->hrir->right_channel, kernels->hrir->channel_length);

        std::transform(
            out_channels->left_channel.begin(), out_channels->left_channel.end(), out_channels->left_channel.begin(),
            [](double x) {
                return std::tanh(x * SOFT_CLIP_FACTOR);
            }
        );

        std::transform(
            out_channels->right_channel.begin(), out_channels->right_channel.end(), out_channels->right_channel.begin(),
            [](double x) {
                return std::tanh(x * SOFT_CLIP_FACTOR);
            }
        );
    }

private:
    size_t chunk;
    double fs;
    Hrir* hkernel;
    Rir* rkernel;
    double morph_direction;
    CurveMode morph_curve;
    double distance;
    OSAConv* hleft_buffer;
    OSAConv* hright_buffer;
    OSAConv* rmono_buffer;
    std::vector<double> mono;
};

#endif
