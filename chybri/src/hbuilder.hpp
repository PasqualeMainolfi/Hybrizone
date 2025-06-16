#ifndef HBUILDER_HPP
#define HBUILDER_HPP

#include "htools.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <format>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

struct HData
{
    std::vector<double> hrtf_mag;
    std::vector<double> hrtf_angle;
    double itd;
    CartesianPoint coord;

    HData() {
        this->hrtf_mag = std::vector<double>();
        this->hrtf_angle = std::vector<double>();
        this->itd = 0.0;
        this->coord = CartesianPoint();
    }

    ~HData() { }
};

struct HInfo
{
    HData* h1;
    HData* h2;
    PolarPoint target;
    HShape shape;
    std::string key;

    HInfo() {
        this->h1 = new HData();
        this->h2 = new HData();
        this->target = PolarPoint();
        this->shape = HShape();
        this->key = "";
    }

    HInfo(const HInfo& other)
    :h1(new HData(*other.h1)), h2(new HData(*other.h2)), target(other.target), shape(other.shape), key(other.key)
    { }

    HInfo(const std::string& k)
    :h1(nullptr), h2(nullptr), target(PolarPoint()), shape(HShape()), key(k)
    { }

    ~HInfo() {
        delete this->h1;
        delete this->h2;
    }
};

class HBuilder
{
public:
    HrirDatasetRead* dataset;
    size_t size;
    size_t hsize;
    double fs;

    HBuilder(const char* dataset_path) {
        this->dataset = new HrirDatasetRead(dataset_path);
        this->fs = (double) this->dataset->get_sample_rate();
        this->iso9613 = nullptr;
        this->db_attenuation = std::vector<double>();
        this->hinfo = nullptr;

        HShape sh = this->dataset->get_shape();
        this->size = sh.rows * sh.cols;
        this->hsize = sh.rows;
        size_t half_size = this->hsize / 2 + 1;

        this->geometric_attenuation = new GeometricAttenuation(this->fs, CHANNELS);
        this->geometric_attenuation->alloc_indexes(this->hsize);

        this->slerp_fft_left_channel = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
        this->slerp_fft_right_channel = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
        this->slerp_ifft_left_channel = std::vector<double>(this->hsize, 0.0);
        this->slerp_ifft_right_channel = std::vector<double>(this->hsize, 0.0);

        this->left_ifft = fftw_plan_dft_c2r_1d(this->hsize, this->slerp_fft_left_channel, this->slerp_ifft_left_channel.data(), FFTW_ESTIMATE);
        this->right_ifft = fftw_plan_dft_c2r_1d(this->hsize, this->slerp_fft_right_channel, this->slerp_ifft_right_channel.data(), FFTW_ESTIMATE);

        this->current_left_channel = std::vector<double>(this->hsize, 0.0);
        this->current_right_channel = std::vector<double>(this->hsize, 0.0);

        this->prev_distance = -1.0;
        this->prev_left_channel = std::vector<double>(this->hsize, 0.0);
        this->prev_right_channel = std::vector<double>(this->hsize, 0.0);

        this->cache = new LRUCache(CACHE_CAPACITY, CacheType::HRIR);

        this->cross_fade_length = static_cast<size_t>(INTERNAL_KERNEL_TRANSITION * this->fs);
        this->cross_fade_coeff_a = std::vector<double>(this->cross_fade_length);
        this->cross_fade_coeff_b = std::vector<double>(this->cross_fade_length);
        for (size_t i = 0; i < this->cross_fade_length; ++i) {
            double alpha_linear =  static_cast<double>(i) / static_cast<double>(this->cross_fade_length - 1);
            double alpha = alpha_linear * M_PI_2;
            this->cross_fade_coeff_a[i] = std::cos(alpha);
            this->cross_fade_coeff_b[i] = std::sin(alpha);
        }

        this->f_shift_itd = std::vector<double>(this->hsize);
        double f_factor = this->fs / static_cast<double>(this->hsize - 1);
        for (size_t i = 0; i < this->hsize; ++i) {
            double f =  static_cast<double>(i) * f_factor;
            this->f_shift_itd[i] = -TWOPI * f;
        }

    }

    void set_air_condition(AirData* air_data) {
        this->iso9613 = new ISO9613Filter(air_data, this->fs);
        this->db_attenuation.resize(NFREQS);
        this->iso9613->get_attenuation_air_absorption(this->db_attenuation.data());
    }

    ~HBuilder() {
        delete this->iso9613;
        delete this->dataset;
        delete this->geometric_attenuation;
        if (this->hinfo) delete this->hinfo;
        delete this->cache;

        fftw_destroy_plan(this->left_ifft);
        fftw_destroy_plan(this->right_ifft);
        fftw_free(this->slerp_fft_left_channel);
        fftw_free(this->slerp_fft_right_channel);
    }

    void hmatching(PolarPoint* p) {
        if (this->hinfo != nullptr) {
            delete this->hinfo;
            this->hinfo = nullptr;
        }

        this->hinfo = new HInfo();
        p->rho = std::max(p->rho, ETA);

        std::string temp_key_ = p->get_polar_key();

        if (this->cache->contains(temp_key_)) {
            this->hinfo->key = temp_key_;
        } else {
            CartesianPoint cart = p->get_cartesian();
            SpatialNeighs sneighs = spatial_match(&cart, this->dataset);

            PolarIndex pindex1 = this->dataset->get_polar_index(sneighs.i0);
            std::string key1 = std::format("{}_{}", pindex1.ele, pindex1.azi);

            PolarIndex pindex2 = this->dataset->get_polar_index(sneighs.i1);
            std::string key2 = std::format("{}_{}", pindex2.ele, pindex2.azi);

            this->hinfo->h1->coord = this->dataset->get_cartesian_reference(sneighs.i0);
            this->dataset->get_itd(&this->hinfo->h1->itd, key1.c_str());
            this->dataset->get_hrir_data(&this->hinfo->h1->hrtf_mag, key1.c_str(), HDataType::HMAG);
            this->dataset->get_hrir_data(&this->hinfo->h1->hrtf_angle, key1.c_str(), HDataType::HANGLE);

            this->hinfo->h2->coord = this->dataset->get_cartesian_reference(sneighs.i1);
            this->dataset->get_itd(&this->hinfo->h2->itd, key2.c_str());
            this->dataset->get_hrir_data(&this->hinfo->h2->hrtf_mag, key2.c_str(), HDataType::HMAG);
            this->dataset->get_hrir_data(&this->hinfo->h2->hrtf_angle, key2.c_str(), HDataType::HANGLE);

            this->hinfo->target = *p;
            this->hinfo->shape = this->dataset->get_shape();
        }
    }

    void build_kernel(Hrir* hrir) {
        hrir->left_channel.resize(this->hsize);
        hrir->right_channel.resize(this->hsize);
        hrir->channel_length = this->hsize;

        std::string temp_key = this->hinfo->key;

        if (this->cache->contains(temp_key)) {
            Hrir* h = (Hrir*) this->cache->get(temp_key);
            memcpy(hrir->left_channel.data(), h->left_channel.data(), sizeof(double) * this->hsize);
            memcpy(hrir->right_channel.data(), h->right_channel.data(), sizeof(double) * this->hsize);
        } else {
            this->hslerp();

            double current_distance = this->hinfo->target.rho;
            this->distance_based(current_distance);

            std::vector<double> target_left = this->current_left_channel;
            std::vector<double> target_right = this->current_right_channel;

            std::vector<double> crossed_left = target_left;
            std::vector<double> crossed_right = target_right;

            if (this->prev_distance != -1.0 && this->prev_distance != current_distance) {
                double d = abs(this->prev_distance - current_distance);
                if (d > MAX_CROSSFADE_DISTANCE) {
                    cross_fade(this->prev_left_channel.data(), crossed_left.data(), this->cross_fade_coeff_a.data(), this->cross_fade_coeff_b.data(), this->cross_fade_length);
                    cross_fade(this->prev_right_channel.data(), crossed_right.data(), this->cross_fade_coeff_a.data(), this->cross_fade_coeff_b.data(), this->cross_fade_length);
                }
            }

            memcpy(hrir->left_channel.data(), crossed_left.data(), sizeof(double) * this->hsize);
            memcpy(hrir->right_channel.data(), crossed_right.data(), sizeof(double) * this->hsize);

            this->prev_left_channel = target_left;
            this->prev_right_channel = target_right;
            this->prev_distance = this->hinfo->target.rho;

            this->cache->put(this->hinfo->target.get_polar_key(), (Hrir*) new Hrir(target_left.data(), target_right.data(), this->hsize));
        }
    }

private:
    GeometricAttenuation* geometric_attenuation;
    ISO9613Filter* iso9613;
    std::vector<double> db_attenuation;
    HInfo* hinfo;
    fftw_complex* slerp_fft_left_channel;
    fftw_complex* slerp_fft_right_channel;
    std::vector<double> slerp_ifft_left_channel;
    std::vector<double> slerp_ifft_right_channel;
    fftw_plan left_ifft;
    fftw_plan right_ifft;
    std::vector<double> current_left_channel;
    std::vector<double> current_right_channel;
    std::vector<double> prev_left_channel;
    std::vector<double> prev_right_channel;
    double prev_distance;
    LRUCache* cache;
    size_t cross_fade_length;
    std::vector<double> cross_fade_coeff_a;
    std::vector<double> cross_fade_coeff_b;
    std::vector<double> f_shift_itd;

    void hslerp() {
        CartesianPoint trg = this->hinfo->target.get_cartesian();
        SlerpCoeff s_coeffs = slerp_coefficients(&trg, &this->hinfo->h1->coord, &this->hinfo->h2->coord);

        double interpolated_itd;
        if (this->hinfo->target.rho < this->dataset->get_source_distance()) {
            interpolated_itd = woodworth_itd3d(this->hinfo->target);
        } else {
            interpolated_itd = this->hinfo->h1->itd * s_coeffs.a + this->hinfo->h2->itd * s_coeffs.b;
        }

        interpolated_itd = -interpolated_itd;

        size_t half_size = this->hsize / 2 + 1;

        std::vector<double> unwrapped_h1_angle = this->hinfo->h1->hrtf_angle;
        std::vector<double> unwrapped_h2_angle = this->hinfo->h2->hrtf_angle;
        unwrap_phase(unwrapped_h1_angle.data(), half_size);
        unwrap_phase(unwrapped_h2_angle.data(), half_size);

        for (size_t i = 0; i < half_size; ++i) {
            size_t j = i * 2;
            double left_mag = this->hinfo->h1->hrtf_mag[j] * s_coeffs.a + this->hinfo->h2->hrtf_mag[j] * s_coeffs.b;
            double right_mag = this->hinfo->h1->hrtf_mag[j + 1] * s_coeffs.a + this->hinfo->h2->hrtf_mag[j + 1] * s_coeffs.b;
            double left_angle = unwrapped_h1_angle[j] * s_coeffs.a + unwrapped_h2_angle[j] * s_coeffs.b;
            double right_angle = unwrapped_h1_angle[j + 1] * s_coeffs.a + unwrapped_h2_angle[j + 1] * s_coeffs.b;

            std::complex<double> lc = std::polar(left_mag, left_angle);
            std::complex<double> rc = std::polar(right_mag, right_angle);

            double ph = this->f_shift_itd[i];
            if (interpolated_itd > 0.0) {
                rc *= std::polar(1.0, ph * interpolated_itd);
            } else {
                lc *= std::polar(1.0, ph * abs(interpolated_itd));
            }

            this->slerp_fft_left_channel[i][0] = lc.real();
            this->slerp_fft_left_channel[i][1] = lc.imag();

            this->slerp_fft_right_channel[i][0] = rc.real();
            this->slerp_fft_right_channel[i][1] = rc.imag();
        }

        fftw_execute(this->left_ifft);
        fftw_execute(this->right_ifft);

        for (size_t i = 0; i < this->hsize; ++i) {
            this->current_left_channel[i] = this->slerp_ifft_left_channel[i] / static_cast<double>(this->hsize);
            this->current_right_channel[i] = this->slerp_ifft_right_channel[i] / static_cast<double>(this->hsize);
        }
    }

    void distance_based(double rho) {
        std::vector<double> left_temp = this->current_left_channel;
        std::vector<double> right_temp = this->current_right_channel;

        this->geometric_attenuation->apply_fractional_delay(left_temp.data(), this->current_left_channel.data(), rho, 0, this->hsize);
        this->geometric_attenuation->apply_fractional_delay(right_temp.data(), this->current_right_channel.data(), rho, 1, this->hsize);

        double rdist = rho - std::max(this->dataset->get_source_distance(), ETA);

        this->iso9613->air_absorption_filter(this->current_left_channel.data(), this->db_attenuation.data(), rdist, this->hsize);
        this->iso9613->air_absorption_filter(this->current_right_channel.data(), this->db_attenuation.data(), rdist, this->hsize);

        double factor = this->geometric_attenuation->calculate_geometric_factor(this->dataset->get_source_distance(), rho);
        factor = factor > 1.0 ? 1.0 : factor;

        for (size_t i = 0; i < this->hsize; ++i) {
            this->current_left_channel[i] *= factor;
            this->current_right_channel[i] *= factor;
        }
    }
};

#endif
