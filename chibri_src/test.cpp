// #include "src/htools.hpp"
#include "src/hbuilder.hpp"
#include "src/htools.hpp"
#include <cstddef>

#define HRIR_DATASET_PATH ("/Users/pm/AcaHub/Coding/BinauralSpatial/data/HRIR-KEMAR_DATASET.h5")

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

    HBuilder hb(HRIR_DATASET_PATH);
    hb.set_air_condition(&air_data);

    PolarPoint target(1.7, 10, 90, AngleMode::DEGREE);

    hb.hmatching(&target);

    double* kernel = nullptr;
    hb.build_kernel(&kernel);

    std::cout << kernel << std::endl;

    free(kernel);



    return 0;
}
