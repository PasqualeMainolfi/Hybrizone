#include "htools.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <xsimd/xsimd.hpp>

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

        if (idx < xsize && x[idx] == value) {
            yout[i] = y[idx];
        } else if (idx > 0 && idx < xsize) {
            double x0 = x[idx - 1];
            double x1 = x[idx];
            double y0 = y[idx -1];
            double y1 = y[idx];

            double t = (value - x0) / (x1 - x0);
            yout[i] = y0 + (y1 - y0) * t;
        } else {
            yout[i] = fill_value ? 0.0 : (idx == 0 ? y[0] : y[xsize - 1]);
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

void cross_fade(double* prev_kernel, double* current_kernel, double* a, double* b, size_t transition_length) {
    for (size_t i = 0; i < transition_length; ++i) {
        current_kernel[i] = a[i] * prev_kernel[i] + b[i] * current_kernel[i];
    }
};

void fft_convolve(std::vector<double, xsimd::aligned_allocator<double>>* buffer, double* x, double* kernel, size_t x_size, size_t k_size, ConvMode conv_mode) {
    size_t conv_size_temp = x_size + k_size - 1;
    size_t conv_size = next_power_of_two(conv_size_temp);
    std::vector<double> x_padded(conv_size, 0.0);
    std::vector<double> k_padded(conv_size, 0.0);

    memcpy(x_padded.data(), x, sizeof(double) * x_size);
    memcpy(k_padded.data(), kernel, sizeof(double) * k_size);

    size_t half_size = conv_size / 2 + 1;
    fftw_complex* xfft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
    fftw_complex* kfft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);

    fftw_plan px = fftw_plan_dft_r2c_1d(conv_size, x_padded.data(), xfft, FFTW_ESTIMATE);
    fftw_plan pk = fftw_plan_dft_r2c_1d(conv_size, k_padded.data(), kfft, FFTW_ESTIMATE);
    fftw_execute(px);
    fftw_execute(pk);

    fftw_complex* ifft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
    for (size_t i = 0; i < half_size; ++i) {
        std::complex<double> xz(xfft[i][0], xfft[i][1]);
        std::complex<double> kz(kfft[i][0], kfft[i][1]);
        std::complex<double> c = xz * kz;
        ifft_in[i][0] = c.real();
        ifft_in[i][1] = c.imag();
    }

    double* ifft_out = (double*) malloc(sizeof(double) * conv_size);
    fftw_plan pifft = fftw_plan_dft_c2r_1d(conv_size, ifft_in, ifft_out, FFTW_ESTIMATE);
    fftw_execute(pifft);

    std::transform(
        ifft_out, ifft_out + conv_size, ifft_out, [&conv_size](double x) {
            return x / static_cast<double>(conv_size);
        }
    );

    size_t length = conv_size;
    size_t offset = 0;
    if (conv_mode == ConvMode::SAME) {
        length = x_size;
        offset = (conv_size_temp - x_size) / 2;
    }

    buffer->resize(length);
    memcpy(buffer->data(), ifft_out + offset, sizeof(double) * length);

    fftw_destroy_plan(px);
    fftw_destroy_plan(pk);
    fftw_destroy_plan(pifft);
    fftw_free(xfft);
    fftw_free(kfft);
    fftw_free(ifft_in);
    free(ifft_out);
}

void intermediate_segment(double* buffer, double* x, double* prev_kernel, double* curr_kernel, size_t ksize, size_t transition_size) {
    for (size_t i = 0; i < transition_size; ++i) {
        double alpha = static_cast<double>(i) / static_cast<double>(transition_size - 1);
        for (size_t j = 0; j < ksize; ++j) {
            double crossed = (1.0 - alpha) * prev_kernel[j] + alpha * curr_kernel[j];
            buffer[i + j] += x[i] * crossed;
        }
    }
}

void apply_intermediate(OSABuffer* osa_buffer, double* x, std::vector<double, xsimd::aligned_allocator<double>> prev_kernel, std::vector<double, xsimd::aligned_allocator<double>> curr_kernel, size_t x_length, size_t prev_kernel_length, size_t curr_kernel_length, size_t transition_length) {
    if (prev_kernel_length <= 0) {
        fft_convolve(&osa_buffer->buffer, x, curr_kernel.data(), x_length, curr_kernel_length, ConvMode::FULL);
        osa_buffer->conv_buffer_size = x_length + curr_kernel_length - 1;
        return;
    }

    std::vector<double, xsimd::aligned_allocator<double>> kprev = prev_kernel;
    std::vector<double, xsimd::aligned_allocator<double>> kcurr = curr_kernel;

    size_t ksize = std::max(prev_kernel_length, curr_kernel_length);
    size_t conv_length = x_length + ksize - 1;
    osa_buffer->buffer.resize(conv_length);
    osa_buffer->conv_buffer_size = conv_length;
    intermediate_segment(osa_buffer->buffer.data(), x, kprev.data(), kcurr.data(), ksize, transition_length);

    size_t x_rest_size = x_length - transition_length;
    std::vector<double, xsimd::aligned_allocator<double>> xtemp(x_rest_size, 0.0);
    memcpy(xtemp.data(), x + transition_length, sizeof(double) * x_rest_size);

    std::vector<double, xsimd::aligned_allocator<double>> rest_part;
    fft_convolve(&rest_part, xtemp.data(), kcurr.data(), x_rest_size, ksize, ConvMode::FULL);

    size_t x_rest_conv_size = x_rest_size + ksize - 1;

    for (size_t i = 0; i < x_rest_conv_size; ++i) {
        osa_buffer->buffer[i + transition_length] += rest_part[i];
    }
}
