// #include "src/htools.hpp"
// #include "src/hbuilder.hpp"
// #include "src/hybri.hpp"
#include "src/htools.hpp"
#include "src/rbuilder.hpp"
#include <cstddef>
#include <chrono>
#include <ratio>

#define HRIR_DATASET_PATH ("/Users/pm/AcaHub/Coding/BinauralSpatial/data/HRIR-KEMAR_DATASET.h5")
#define RIR_DATASET_PATH ("/Users/pm/AcaHub/Coding/BinauralSpatial/data/RIR-MIT_SURVEY.h5")
#define FS (44100.0)
#define CHUNK (2048)

int main(void) {
    // HRIR dataset test
    // HrirDatasetRead hrir(HRIR_DATASET_PATH);

    // double* temp_hrir = nullptr;
    // hrir.get_hrir_data(&temp_hrir, "10:90", HDataType::HANGLE);
    // std::cout << temp_hrir << std::endl;
    // free(temp_hrir);

    // CartesianPoint c = hrir.get_cartesian_reference(10);
    // std::cout << "CARTESIAN REFERENCE: " << c.x << " : " << c.y << " : " << c.z << std::endl;

    // PolarIndex  p = hrir.get_polar_index(10);
    // std::cout << "POLAR REFERENCE: " << p.ele << " : " << p.azi << std::endl;

    // double source_distance = hrir.get_source_distance();
    // std::cout << "SOURCE DISTANCE: " << source_distance << std::endl;

    // double fs = hrir.get_sample_rate();
    // std::cout << "FS: " << fs << std::endl;

    // HShape shape = hrir.get_shape();
    // std::cout << "SHAPE: " << shape.rows << " : " << shape.cols << std::endl;

    // HBUILDER test

    AirData air_data;
    std::cout << air_data.kelvin << std::endl;

    // HBuilder hb(HRIR_DATASET_PATH);
    // hb.set_air_condition(&air_data);

    // Hrir* kernel = new Hrir();
    // double azi = 0.0;
    // while (azi < 360.0) {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     PolarPoint target(3.0, 10.7, azi, AngleMode::DEGREE);

    //     hb.hmatching(&target);

    //     hb.build_kernel(kernel);

    //     // std::cout << kernel->left_channel<< std::endl;
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double, std::milli> elapsed = end - start;
    //     std::cout << "[INFO] TIME: " << elapsed << std::endl;
    //     azi += 0.1;
    // }

    // delete kernel;

    // Hybrizone hybri(HRIR_DATASET_PATH, CHUNK, FS);
    // hybri.set_air_condition(air_data);

    // PolarPoint target(3.0, 10.7, 90.0, AngleMode::DEGREE);
    // hybri.set_target_position(target);
    // hybri.generate_kernels();
    //

    RBuilder rb(RIR_DATASET_PATH, 1.7, FS);

    return 0;
}
