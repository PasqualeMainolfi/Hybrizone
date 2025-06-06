#include "htools.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

void lerp(double* x, double* y, double* xnew, double* yout, size_t xsize, size_t xnew_size, bool fill_value) {
    for (size_t i = 0; i < xnew_size; ++i) {
        double value = xnew[i];

        if (value < x[0]) {
            yout[i] = fill_value ? 0.0 : y[0];
            continue;
        }

        if (value > x[xsize - 1]) {
            yout[i] = fill_value ? 0.0 : y[xsize - 1];
            continue;
        }

        auto it = std::lower_bound(x, x + xsize, value);
        size_t idx = it - x;

        if (x[idx] == value || idx == 0) {
            yout[i] = y[idx];
        } else {
            double x0 = x[idx - 1];
            double x1 = x[idx];
            double y0 = y[idx -1];
            double y1 = y[idx];

            double t = (value - x0) / (x1 - x0);
            yout[i] = y0 + (y1 - y0) * t;
            // std::cout << yout[i] << std::endl;
        }
    }
}

double woodworth_itd3d(const PolarPoint& p) {
    double sin_theta = sin(-p.theta);
    double cos_phi = cos(p.phi);
    double svalue = p.rho * p.rho + HEAD_RADIUS * HEAD_RADIUS - 2 * HEAD_RADIUS * p.rho * sin_theta * cos_phi;
    double num = p.rho + HEAD_RADIUS * sin_theta * cos_phi - sqrt(svalue);
    return num / SOUND_SPEED;
}

SpatialNeighs spatial_match(CartesianPoint* target, HrirDatasetRead* dataset) {
    SpatialNeighs sn;
    sn.a_dist = std::numeric_limits<double>::infinity();
    sn.b_dist = std::numeric_limits<double>::infinity();
    sn.i0 = 0;
    sn.i1 = 0;

    for (size_t i = 0; i < dataset->get_dataset_size(); ++i) {
        CartesianPoint source = dataset->get_cartesian_reference(static_cast<int>(i));
        double dist = target->get_distance(&source);
        if (dist < sn.a_dist) {
            sn.b_dist = sn.a_dist;
            sn.i1 = sn.i0;
            sn.a_dist = dist;
            sn.i0 = i;
        } else {
            if (dist < sn.b_dist) {
                sn.b_dist = dist;
                sn.i1 = i;
            }
        }
    }
    return sn;
};

void cross_fade(double* prev_kernel, double* current_kernel, size_t transition_length) {
    for (size_t i = 0; i < transition_length; ++i) {
        double alpha_linear =  i / static_cast<double>(transition_length - 1);
        double alpha = alpha_linear * M_PI_2;
        double a = cos(alpha);
        double b = sin(alpha);
        current_kernel[i] = a * prev_kernel[i] + b * current_kernel[i];
    }
};
