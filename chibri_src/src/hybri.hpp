#ifndef HYBRI_HPP
#define HYBRI_HPP

#include "hbuilder.hpp"
#include "htools.hpp"
#include "rbuilder.hpp"
#include <cstddef>
#include <cstdlib>


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
        delete hrir;
        if (rir) free(rir);
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

        if (rir_dataset_path) {
            this->rbuilder = new RBuilder(rir_dataset_path, this->hbuilder->dataset->get_source_distance(), sample_rate);
            this->rkernel = new Rir();
        } else {
            this->rbuilder = nullptr;
            this->rkernel = nullptr;
        }

        this->morph_direction = 0.0;
        this->morph_curve = CurveMode::LINEAR;

        this->distance = 0.0;

        if (this->fs != this->hbuilder->fs) {
            std::cerr << "[ERROR] Sample rate must be equal to HRIR's sample rate!" << std::endl;
            std::exit(1);
        }

    }

    ~Hybrizone() {
        delete this->hbuilder;
        delete this->hkernel;

        if (this->rbuilder) {
            delete this->rbuilder;
            delete this->rkernel;
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
        if (this->rbuilder) {
            this->rbuilder->build_hybrid_rir(this->rkernel, this->morph_direction, this->morph_curve, this->distance);
            kernels->rir = new Rir(*this->rkernel);
        }
    }

private:
    size_t chunk;
    double fs;
    Hrir* hkernel;
    Rir* rkernel;
    double morph_direction;
    CurveMode morph_curve;
    double distance;
};

#endif
