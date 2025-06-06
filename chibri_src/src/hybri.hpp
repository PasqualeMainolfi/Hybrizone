#ifndef HYBRI_HPP
#define HYBRI_HPP

#include "hbuilder.hpp"
#include "htools.hpp"
#include <cstddef>
#include <cstdlib>

class Hybrizone
{
public:
    HBuilder* hbuilder;

    Hybrizone(const char* hrir_dataset_path, size_t chunk_size, double sample_rate)
    : chunk(chunk_size), fs(sample_rate)
    {
        this->hbuilder = new HBuilder(hrir_dataset_path);
        this->hkernel = new Hrir();

        if (this->fs != this->hbuilder->fs) {
            std::cerr << "[ERROR] Sample rate must be equal to HRIR's sample rate!" << std::endl;
            std::exit(1);
        }

    }

    ~Hybrizone() {
        delete this->hbuilder;
        delete this->hkernel;
    }

    void set_air_condition(AirData air_data) {
        this->hbuilder->set_air_condition(&air_data);
    }

    void set_target_position(PolarPoint p) {
        this->hbuilder->hmatching(&p);
    }

    void generate_kernels() {
        this->hbuilder->build_kernel(this->hkernel);
    }

private:
    size_t chunk;
    double fs;
    Hrir* hkernel;
};

#endif
