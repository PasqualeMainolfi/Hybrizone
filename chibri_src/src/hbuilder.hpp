#ifndef HBUILDER_HPP
#define HBUILDER_HPP

#include "htools.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <cstddef>
#include <cstdlib>
#include <string>

struct HData
{
    double* hrtf_mag;
    double* hrtf_angle;
    double* itd;
    CartesianPoint coord;

    HData() {
        this->hrtf_mag = nullptr;
        this->hrtf_angle = nullptr;
        this->itd = nullptr;
        this->coord = CartesianPoint();
    }

    ~HData() {
        free(this->hrtf_mag);
        free(this->hrtf_angle);
        free(this->itd);
    }
};

struct HInfo
{
    HData* h1;
    HData* h2;
    PolarPoint target;
    HShape shape;

    HInfo() {
        this->h1 = new HData();
        this->h2 = new HData();
        this->target = PolarPoint();
        this->shape = HShape();
    }

    ~HInfo() {
        delete this->h1;
        delete this->h2;
    }
};

class HBuilder
{
public:
    HBuilder(const char* dataset_path) {
        this->dataset = new HrirDatasetRead(dataset_path);
        this->geometric_attenuation = new GeometricAttenuation(this->dataset->get_sample_rate(), CHANNELS);
        this->iso9613 = nullptr;
        this->db_attenuation = nullptr;
        this->hinfo = new HInfo();

        HShape sh = this->dataset->get_shape();
        this->size = sh.rows * sh.cols;
        this->hsize = sh.rows;
        size_t half_size = sh.rows / 2 + 1;

        this->slerp_fft_left_channel = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
        this->slerp_fft_right_channel = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
        this->slerp_ifft_left_channel = (double*) fftw_malloc(sizeof(double) * this->hsize);
        this->slerp_ifft_right_channel = (double*) fftw_malloc(sizeof(double) * this->hsize);
        this->left_ifft = fftw_plan_dft_c2r_1d(this->size, this->slerp_fft_left_channel, this->slerp_ifft_left_channel, FFTW_MEASURE);
        this->right_ifft = fftw_plan_dft_c2r_1d(this->size, this->slerp_fft_right_channel, this->slerp_ifft_right_channel, FFTW_MEASURE);

        this->current_left_channel = (double*) malloc(sizeof(double) * sh.rows);
        this->current_right_channel = (double*) malloc(sizeof(double) * sh.rows);

        this->prev_distance = -1.0;
        this->prev_left_channel = (double*) malloc(sizeof(double) * this->hsize);
        this->prev_right_channel = (double*) malloc(sizeof(double) * this->hsize);
    }

    void set_air_condition(AirData* air_data) {
        this->iso9613 = new ISO9613Filter(air_data, this->dataset->get_sample_rate());
        this->db_attenuation = (double*) malloc(sizeof(double) * NFREQS);
        this->iso9613->get_attenuation_air_absorption(this->db_attenuation);
    }

    ~HBuilder() {
        if (this->db_attenuation) {
            free(this->db_attenuation);
            delete this->iso9613;
        }

        delete this->geometric_attenuation;
        delete this->hinfo;

        fftw_destroy_plan(this->left_ifft);
        fftw_destroy_plan(this->right_ifft);
        fftw_free(this->slerp_fft_left_channel);
        fftw_free(this->slerp_fft_right_channel);
        fftw_free(this->slerp_ifft_left_channel);
        fftw_free(this->slerp_ifft_right_channel);
        free(this->current_left_channel);
        free(this->current_right_channel);
        free(this->prev_left_channel);
        free(this->prev_right_channel);
    }

    void hmatching(PolarPoint* p) {
        p->rho = std::max(p->rho, ETA);
        CartesianPoint cart = p->get_cartesian();
        SpatialNeighs sneighs = spatial_match(&cart, this->dataset);

        PolarIndex pindex1 = this->dataset->get_polar_index(sneighs.i0);
        std::string key1 = std::format("{}:{}", pindex1.ele, pindex1.azi);

        PolarIndex pindex2 = this->dataset->get_polar_index(sneighs.i1);
        std::string key2 = std::format("{}:{}", pindex2.ele, pindex2.azi);

        this->hinfo->h1->coord = this->dataset->get_cartesian_reference(sneighs.i0);
        this->dataset->get_hrir_data(&this->hinfo->h1->itd, key1.c_str(), HDataType::HITD);
        this->dataset->get_hrir_data(&this->hinfo->h1->hrtf_mag, key1.c_str(), HDataType::HMAG);
        this->dataset->get_hrir_data(&this->hinfo->h1->hrtf_angle, key1.c_str(), HDataType::HANGLE);

        this->hinfo->h2->coord = this->dataset->get_cartesian_reference(sneighs.i1);
        this->dataset->get_hrir_data(&this->hinfo->h2->itd, key2.c_str(), HDataType::HITD);
        this->dataset->get_hrir_data(&this->hinfo->h2->hrtf_mag, key2.c_str(), HDataType::HMAG);
        this->dataset->get_hrir_data(&this->hinfo->h2->hrtf_angle, key2.c_str(), HDataType::HANGLE);

        this->hinfo->target = *p;
        this->hinfo->shape = this->dataset->get_shape();
    }

    void hslerp() {

        CartesianPoint trg = this->hinfo->target.get_cartesian();

        CartesianPoint trgnorm = trg.normalize();
        CartesianPoint h1norm = this->hinfo->h1->coord.normalize();
        CartesianPoint h2norm = this->hinfo->h2->coord.normalize();

        double dot = h1norm.x * h2norm.x + h1norm.y * h2norm.y + h1norm.z * h2norm.z;
        double omega = std::acos(dot);
        omega = std::min(std::max(omega, -1.0), 1.0);

        double interpolated_itd;

        double a = 0.0;
        double b = 1.0;
        if (omega > 0.0) {
            double tdot1 = trgnorm.x * h1norm.x + trgnorm.y * h1norm.y + trgnorm.z * h1norm.z;
            double omega1 = acos(tdot1);
            omega1 = std::min(std::max(omega1, -1.0), 1.0);
            double tdot2 = trgnorm.x * h2norm.x + trgnorm.y * h2norm.y + trgnorm.z * h2norm.z;
            double omega2 = acos(tdot2);
            omega2 = std::min(std::max(omega2, -1.0), 1.0);
            double sin_omega = sin(omega);
            double alpha = omega1 / (omega1 + omega2);
            a = (sin((1 - alpha) * omega) / sin_omega);
            b = (sin(alpha * omega) / sin_omega);
        }

        if (this->hinfo->target.rho < this->dataset->get_source_distance()) {
            interpolated_itd = woodworth_itd3d(this->hinfo->target);
        } else {
            interpolated_itd = (*this->hinfo->h1->itd) * a + (*this->hinfo->h2->itd) * b;
        }

        size_t half_size = this->hsize / 2 + 1;

        memset(this->slerp_fft_left_channel, 0, sizeof(fftw_complex) * half_size);
        memset(this->slerp_fft_right_channel, 0, sizeof(fftw_complex) * half_size);
        memset(this->slerp_ifft_left_channel, 0, sizeof(double) * this->hsize);
        memset(this->slerp_ifft_right_channel, 0, sizeof(double) * this->hsize);

        unwrap_phase(this->hinfo->h1->hrtf_angle, half_size);
        unwrap_phase(this->hinfo->h2->hrtf_angle, half_size);

        for (size_t i = 0; i < half_size; ++i) {
            size_t left_index = i * 2;
            size_t right_index = left_index + 1;
            double left_mag = this->hinfo->h1->hrtf_mag[left_index] * a + this->hinfo->h2->hrtf_mag[left_index] * b;
            double right_mag = this->hinfo->h1->hrtf_mag[right_index] * a + this->hinfo->h2->hrtf_mag[right_index] * b;
            double left_angle = this->hinfo->h1->hrtf_angle[left_index] * a + this->hinfo->h2->hrtf_angle[left_index] * b;
            double right_angle = this->hinfo->h1->hrtf_angle[right_index] * a + this->hinfo->h2->hrtf_angle[right_index] * b;
            this->slerp_fft_left_channel[i][0] = left_mag * cos(left_angle);
            this->slerp_fft_left_channel[i][1] = left_mag * sin(left_angle);
            this->slerp_fft_right_channel[i][0] = right_mag * cos(right_angle);
            this->slerp_fft_right_channel[i][1] = right_mag * sin(right_angle);
        }

        interpolated_itd = -interpolated_itd;

        for (size_t i = 0; i < half_size; ++i) {
            double f = i * this->dataset->get_sample_rate() / this->hsize; // maybe multiply by 2 (f problem fftw)
            if (interpolated_itd > 0.0) {
                double phase_shift = -TWOPI * f * interpolated_itd;
                this->slerp_fft_right_channel[i][0] *= cos(phase_shift);
                this->slerp_fft_right_channel[i][1] *= sin(phase_shift);
            } else {
                double phase_shift = -TWOPI * f * abs(interpolated_itd);
                this->slerp_fft_left_channel[i][0] *= cos(phase_shift);
                this->slerp_fft_left_channel[i][1] *= sin(phase_shift);
            }
        }

        fftw_execute(this->left_ifft);
        fftw_execute(this->right_ifft);

        for (size_t i = 0; i < half_size; ++i) {
            this->current_left_channel[i] = this->slerp_ifft_left_channel[i] / this->hsize;
            this->current_right_channel[i] = this->slerp_ifft_right_channel[i] / this->hsize;
        }
    }

    void distance_based(double rho) {
        double factor = this->geometric_attenuation->calculate_geometric_factor(this->dataset->get_source_distance(), rho);
        double* left_temp = (double*) malloc(sizeof(double) * this->hsize);
        double* right_temp = (double*) malloc(sizeof(double) * this->hsize);

        memcpy(left_temp, this->current_left_channel, sizeof(double) * this->hsize);
        memcpy(right_temp, this->current_right_channel, sizeof(double) * this->hsize);

        this->geometric_attenuation->apply_fractional_delay(this->current_left_channel, left_temp, rho, 0, this->hsize);
        this->geometric_attenuation->apply_fractional_delay(this->current_right_channel, right_temp, rho, 0, this->hsize);

        memcpy(this->current_left_channel, left_temp, sizeof(double) * this->hsize);
        memcpy(this->current_right_channel, right_temp, sizeof(double) * this->hsize);

        double rdist = rho - std::max(this->dataset->get_source_distance(), ETA);

        this->iso9613->air_absorption_filter(this->current_left_channel, this->db_attenuation, rdist, this->hsize);
        this->iso9613->air_absorption_filter(this->current_right_channel, this->db_attenuation, rdist, this->hsize);

        for (size_t i = 0; i < this->hsize; ++i) {
            this->current_left_channel[i] *= factor;
            this->current_right_channel[i] *= factor;
        }

        free(left_temp);
        free(right_temp);
    }

    void build_kernel(double** kernel) {
        *kernel = (double*) malloc(sizeof(double) * this->size);
        this->hslerp();
        double current_distance = this->hinfo->target.rho;
        this->distance_based(current_distance);
        double* left_temp = (double*) malloc(sizeof(double) * this->hsize);
        double* right_temp = (double*) malloc(sizeof(double) * this->hsize);

        memcpy(left_temp, this->current_left_channel, sizeof(double) * this->hsize);
        memcpy(right_temp, this->current_right_channel, sizeof(double) * this->hsize);

        if (this->prev_distance != -1.0 && this->prev_distance != current_distance) {
            double d = abs(this->prev_distance - current_distance);
            if (d < MAX_CROSSFADE_DISTANCE) {
                size_t tlength = static_cast<size_t>(INTERNAL_KERNEL_TRANSITION * this->dataset->get_sample_rate());
                cross_fade(this->prev_left_channel, this->current_left_channel, tlength);
                cross_fade(this->prev_right_channel, this->current_right_channel, tlength);
            }
        }

        memcpy(this->prev_left_channel, left_temp, this->hsize);
        memcpy(this->prev_right_channel, right_temp, this->hsize);

        this->prev_distance = current_distance;

        for (size_t i = 0; i < this->hsize; ++i) {
            (*kernel)[i * 2] = this->current_left_channel[i];
            (*kernel)[i * 2 + 1] = this->current_right_channel[i];
        }

        free(left_temp);
        free(right_temp);
    }

private:
    HrirDatasetRead* dataset;
    GeometricAttenuation* geometric_attenuation;
    ISO9613Filter* iso9613;
    double* db_attenuation;
    HInfo* hinfo;
    fftw_complex* slerp_fft_left_channel;
    fftw_complex* slerp_fft_right_channel;
    double* slerp_ifft_left_channel;
    double* slerp_ifft_right_channel;
    fftw_plan left_ifft;
    fftw_plan right_ifft;
    double* current_left_channel;
    double* current_right_channel;
    double* prev_left_channel;
    double* prev_right_channel;
    double prev_distance;
    size_t size;
    size_t hsize;
};

#endif
