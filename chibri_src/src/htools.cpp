#include "htools.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
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

void fft_convolve(double** buffer, double* x, double* kernel, size_t x_size, size_t k_size, ConvMode conv_mode) {
    size_t conv_size = x_size + k_size - 1;
    double* x_padded = (double*) malloc(sizeof(double) * conv_size);
    double* k_padded = (double*) malloc(sizeof(double) * conv_size);
    memset(x_padded, 0, sizeof(double) * conv_size);
    memset(k_padded, 0, sizeof(double) * conv_size);

    memcpy(x_padded, x, sizeof(double) * x_size);
    memcpy(k_padded, kernel, sizeof(double) * k_size);

    size_t half_size = conv_size / 2 + 1;
    fftw_complex* xfft = (fftw_complex*) fftw_malloc(sizeof(double) * half_size);
    fftw_complex* kfft = (fftw_complex*) fftw_malloc(sizeof(double) * half_size);

    fftw_plan px = fftw_plan_dft_r2c_1d(conv_size, x_padded, xfft, FFTW_MEASURE);
    fftw_plan pk = fftw_plan_dft_r2c_1d(conv_size, k_padded, kfft, FFTW_MEASURE);

    fftw_complex* ifft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
    for (size_t i = 0; i < half_size; ++i) {
        std::complex<double> xz(xfft[i][0], xfft[i][1]);
        std::complex<double> kz(kfft[i][0], kfft[i][1]);
        std::complex<double> c = xz * kz;
        ifft_in[i][0] = c.real();
        ifft_in[i][1] = c.imag();
    }

    double* ifft_out = (double*) malloc(sizeof(double) * conv_size);
    fftw_plan pifft = fftw_plan_dft_c2r_1d(conv_size, ifft_in, ifft_out, FFTW_MEASURE);

    size_t length = conv_size;
    size_t offset = 0;
    if (conv_mode == ConvMode::SAME) {
        length = x_size >= k_size ? x_size : k_size;
        offset = (conv_size - length) / 2;
    }

    *buffer = (double*) malloc(sizeof(double) * length);
    memcpy(*buffer, ifft_out + offset, sizeof(double) * length);

    fftw_destroy_plan(px);
    fftw_destroy_plan(pk);
    fftw_destroy_plan(pifft);
    fftw_free(xfft);
    fftw_free(kfft);
    fftw_free(ifft_in);
    free(x_padded);
    free(k_padded);
    free(ifft_out);

}
