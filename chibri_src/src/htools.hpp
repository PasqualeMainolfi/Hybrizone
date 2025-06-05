#ifndef HTOOLS_HPP
#define HTOOLS_HPP

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fftw3.h>

#define P_REF (101325.0)
#define T0 (293.15)
#define NFREQS (256)
#define MAX_DELAY_SEC (1.0)
#define SOUND_SPEED (343.3)
#define SLEW_RATE (0.01)
#define HEAD_RADIUS (0.0875)
#define ETA (HEAD_RADIUS + 0.01)
#define GAMMA (0.7)
#define TWOPI (2.0 * M_PI)
#define MAX_CROSSFADE_DISTANCE (0.5)
#define INTERNAL_KERNEL_TRANSITION  (0.003)
#define CHANNELS (2)

inline double deg2rad(double deg_value) {
    return deg_value * M_PI / 180.0;
}

inline double rad2deg(double rad_value) {
    return 180.0 * rad_value / M_PI;
}

inline void unwrap_phase(double* x, size_t length) {
    double* buffer = (double*) malloc(sizeof(double) * 2 * length);
    buffer[0] = x[0];
    buffer[1] = x[1];
    double left_correction = 0.0;
    double right_correction = 0.0;
    for (size_t i = 1; i < length; ++i) {
        size_t left_index_prev = (i - 1) * 2;
        size_t left_index_curr = i * 2;
        double left_diff = x[left_index_curr] - x[left_index_prev];
        double right_diff = x[left_index_curr + 1] - x[left_index_prev + 1];
        double left_jump = std::abs(left_diff) <= M_PI ? 0.0 : std::round(left_diff / TWOPI);
        double right_jump = std::abs(right_diff) <= M_PI ? 0.0 : std::round(right_diff / TWOPI);
        left_correction -= left_jump * TWOPI;
        right_correction -= right_jump * TWOPI;
        buffer[left_index_curr] = x[left_index_curr] + left_correction;
        buffer[left_index_curr + 1] = x[left_index_curr + 1] + right_correction;
    }
    memcpy(x, buffer, sizeof(double) * 2 * length);
    free(buffer);
}

enum HDataType
{
    HTIME,
    HMAG,
    HANGLE,
    HITD
};

enum AngleMode
{
    RADIANS,
    DEGREE
};

struct HShape
{
    size_t rows;
    size_t cols;

    HShape() = default;
    HShape(size_t rows_, size_t cols_)
    : rows(rows_), cols(cols_)
    { }
};

struct CartesianPoint
{
public:
    double x;
    double y;
    double z;

    CartesianPoint() = default;
    CartesianPoint(double x_, double y_, double z_)
    :x(x_), y(y_), z(z_)
    { }

    double get_distance(CartesianPoint* p) {
        double x_temp = this->x - p->x;
        double y_temp = this->y - p->y;
        double z_temp = this->z - p->z;
        return sqrt(x_temp * x_temp + y_temp * y_temp + z_temp * z_temp);
    }

    CartesianPoint normalize() {
        double norm = sqrt(this->x * this->x + this->y * this->y + this->z * this->z);

        CartesianPoint pn(0.0, 0.0, 0.0);
        if (norm > 0.0) {
            pn.x = this->x / norm;
            pn.y = this->y / norm;
            pn.z = this->z / norm;
        }

        return pn;
    }
};

struct PolarIndex
{
public:
    int ele;
    int azi;
    PolarIndex(int ele_, int azi_)
    : ele(ele_), azi(azi_)
    { }
};

struct PolarPoint
{
public:
    double rho;
    double phi;
    double theta;
    AngleMode mode;

    PolarPoint() = default;
    PolarPoint(double rho_, double phi_, double theta_, AngleMode angle_mode)
    : rho(rho_), mode(angle_mode)
    {
        this->phi = phi_;
        this->theta = theta_;

        if (angle_mode == AngleMode::DEGREE) {
            this->phi = deg2rad(phi_);
            this->theta = deg2rad(theta_);
        }
    }

    CartesianPoint get_cartesian() {
        double x = sin(this->theta) * cos(this->phi);
        double y = cos(this->theta) * cos(this->phi);
        double z = sin(this->phi);

        x = x * 0.99;
        y = y * 0.99 + 0.01;
        z = z * 0.99;

        return CartesianPoint(x, y, z);
    }

};

class HrirDatasetRead
{
    H5::H5File hdata;
    H5::Group ghrir;
    H5::Group ghrtf;
    H5::Group gitd;
    int* polar_index;
    double* regular_coords;
    HShape hrir_shape;
    double source_distance;
    double sample_rate;
    size_t dataset_size;


public:
    HrirDatasetRead() = default;
    HrirDatasetRead(const char* dataset_path) {
        this->hdata = H5::H5File(dataset_path, H5F_ACC_RDONLY);
        this->ghrir = this->hdata.openGroup("hrir");
        this->ghrtf = this->hdata.openGroup("hrir_fft");
        this->gitd = this->hdata.openGroup("hrir_itd");

        // read polar indexes
        H5::DataSet pindex = this->hdata.openDataSet("polar_index");
        H5::DataSpace pindex_dataspace = pindex.getSpace();
        hsize_t dims[2];
        pindex_dataspace.getSimpleExtentDims(dims);
        this->polar_index = (int*) malloc(sizeof(int) * dims[0] * dims[1]);
        pindex.read(this->polar_index, H5::PredType::NATIVE_INT);

        this->dataset_size = dims[0];

        // read cartesian coords
        H5::DataSet cindex = this->hdata.openDataSet("regular_coords");
        H5::DataSpace cindex_dataspace = cindex.getSpace();
        cindex_dataspace.getSimpleExtentDims(dims);
        this->regular_coords = (double*) malloc(sizeof(double) * dims[0] * dims[1]);
        cindex.read(this->regular_coords, H5::PredType::NATIVE_DOUBLE);

        // read attributes
        H5::Attribute sd = this->hdata.openAttribute("source_distance");
        sd.read(H5::PredType::NATIVE_DOUBLE, &this->source_distance);
        H5::Attribute fs = this->hdata.openAttribute("fs");
        fs.read(H5::PredType::NATIVE_DOUBLE, &this->sample_rate);
        H5::Attribute shape = this->hdata.openAttribute("hrir_shape");
        H5::DataSpace shape_space = shape.getSpace();
        shape_space.getSimpleExtentDims(dims);
        size_t* temp_shape = (size_t*) malloc(sizeof(size_t) * dims[0]);
        shape.read(H5::PredType::NATIVE_HSIZE, temp_shape);
        this->hrir_shape = HShape(temp_shape[0], temp_shape[1]);
        free(temp_shape);
    }

    ~HrirDatasetRead() {
        free(this->polar_index);
        free(this->regular_coords);
    }

    // function can return hrir, itd, hrtf mag and angle
    void get_hrir_data(double** buffer, const char* key, HDataType data_type) {
        H5::DataSet h;
        H5::Group fft_group;
        hsize_t dims[2];
        switch (data_type) {
            case HDataType::HTIME:
                h = this->ghrir.openDataSet(key);
                break;
            case HDataType::HMAG:
                fft_group = this->ghrtf.openGroup(key);
                h = fft_group.openDataSet("mag");
                break;
            case HDataType::HANGLE:
                fft_group = this->ghrtf.openGroup(key);
                h = fft_group.openDataSet("angle");
                break;
            case HDataType::HITD:
                h = this->gitd.openDataSet(key);
                break;
        }

        H5::DataSpace dataspace = h.getSpace();
        int ndims = dataspace.getSimpleExtentDims(dims);
        int d = (ndims > 0) ? ndims : 1;
        size_t buffer_size = ndims == 2 ? dims[0] * dims[1] : d;
        *buffer = (double*) malloc(sizeof(double) * buffer_size);

        if (*buffer == nullptr && buffer_size > 0) {
            std::cerr << "[ERROR] Bad alloc in HRIR allocation!";
            std::exit(1);
        }

        h.read(*buffer, H5::PredType::NATIVE_DOUBLE);
    }

    CartesianPoint get_cartesian_reference(int index) {
        double x = this->regular_coords[index * 3];
        double y = this->regular_coords[index * 3 + 1];
        double z = this->regular_coords[index * 3 + 2];
        return CartesianPoint(x, y, z);
    }

    PolarIndex get_polar_index(int index) {
        double azi = this->polar_index[index * 2];
        double ele = this->polar_index[index * 2 + 1];
        return PolarIndex(ele, azi);
    }

    double get_source_distance() {
        return this->source_distance;
    }
    double get_sample_rate() {
        return this->sample_rate;
    }
    HShape get_shape() {
        return this->hrir_shape;
    }

    size_t get_dataset_size() {
        return this->dataset_size;
    }

};

struct AirData
{
public:
  double kelvin;
  double rh;
  double p_atm;

  AirData()
  : kelvin(293.15), rh(50.0), p_atm(1.0)
  { }

  AirData(double temperature, double r_humidity, double pressure)
  : rh(r_humidity)
  {
      this->kelvin = temperature + 273.15;
      this->p_atm = pressure / P_REF;
  }
};

// interpolator
void lerp(double* x, double* y, double* xnew, double* yout, size_t xsize, size_t xnew_size, bool fill_value);

// itd calculation
double woodworth_itd3d(const PolarPoint& p);


class ISO9613Filter
{
public:
    AirData* air_data;

    ISO9613Filter(AirData* air_condition, double fs)
    : air_data(air_condition)
    {
        this->sample_rate = fs;
        this->frequencies = (double*) malloc(sizeof(double) * NFREQS);
        this->fnorm = (double*) malloc(sizeof(double) * NFREQS);

        double fstep = (this->sample_rate / 2.0) / static_cast<double>(NFREQS - 1);
        for (size_t i = 0; i < NFREQS; ++i) {
            double value = i * fstep;
            this->frequencies[i] = value;
            this->fnorm[i] = value / (this->sample_rate / 2.0);
        }

        this->fresp = nullptr;
        this->ftemp = nullptr;
        this->fftout = nullptr;
    }

    ~ISO9613Filter() {
        free(this->frequencies);
        free(this->fnorm);
        free(this->fresp);
        free(this->ftemp);
        fftw_free(this->fftout);
    }

    void get_attenuation_air_absorption(double* alpha) {
        double p_sat = P_REF * (pow(10.0, (-6.8346 * pow((273.16 / this->air_data->kelvin), 1.261) + 4.6151)));
        double h = this->air_data->rh * (p_sat / (this->air_data->p_atm * P_REF));
        double tr = this->air_data->kelvin / T0;
        double tr_pos = pow(tr, 0.5);
        double tr_neg1 = pow(tr, -0.5);
        double tr_neg2 = pow(tr, -2.5);


        double f_rO = this->air_data->p_atm * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));

        double f_rN = this->air_data->p_atm * tr_neg1 * (9.0 + 280.0 * h * exp(-4.17 * (pow(tr, -1.0 / 3.0) - 1.0)));
        double f_rN_square = f_rN * f_rN;

        double alpha_classical = 1.84e-11 * (1.0 / this->air_data->p_atm) * tr_pos;

        double ao_term = 0.01275 * exp(-2239.1 / this->air_data->kelvin);
        double an_term = 0.1068 * exp(-3352.0 / this->air_data->kelvin);

        for (size_t i = 0; i < NFREQS; ++i) {
            double fsquare = pow(this->frequencies[i], 2.0);
            double alpha_oxygen = ao_term * pow(f_rO + (fsquare / f_rO), -1.0);
            double alpha_nitrogen = an_term * pow(f_rN + (f_rN_square + fsquare), -1.0);
            double alpha_term = alpha_classical + tr_neg2 * (alpha_oxygen + alpha_nitrogen);
            alpha[i] = 8.686 * fsquare * alpha_term;
        }
    }

    void air_absorption_filter(double* frame, double* alpha, double distance, size_t frame_size) {
        double rdist = std::max(0.0, distance);
        for (size_t i = 0; i < NFREQS; ++i) {
            double db_attenuation = alpha[i] * rdist;
            alpha[i] = pow(10.0, -db_attenuation / 20.0);
        }
        alpha[NFREQS - 1] = 0.0;
        this->multiband_fft_filter(frame, alpha, frame_size);
    }

private:
    double sample_rate;
    double* frequencies;
    double* fnorm;
    double* fresp;
    double* ftemp;
    fftw_complex* fftout;

    // multiband filter (apply on a single channel Left and Right)
    void multiband_fft_filter(double* frame, double* alpha, size_t frame_size) {
        size_t half_size = static_cast<size_t>(frame_size / 2 + 1);
        double step = 1.0 / half_size;

        double* temp = (double*) realloc(this->ftemp, sizeof(double) * half_size);

        if (!temp) {
            std::cerr << "[ERROR] Failed realloc in multiband filter" << std::endl;
            return;
        }

        this->ftemp = temp;
        for (size_t i = 0; i < half_size; ++i) {
            this->ftemp[i] = step * (double) i;
        }

        double* resp = (double*) realloc(this->fresp, sizeof(double) * half_size);

        if (!resp) {
            std::cerr << "[ERROR] Failed realloc in multiband filter" << std::endl;
            return;
        }

        this->fresp = resp;
        lerp(frame, alpha, this->ftemp, this->fresp, frame_size, half_size, false);

        fftw_complex* temp_fft = (fftw_complex*) realloc(this->fftout, sizeof(fftw_complex) * half_size);
        this->fftout = temp_fft;
        fftw_plan fft_plan = fftw_plan_dft_r2c_1d(frame_size, frame, this->fftout, FFTW_MEASURE);
        fftw_execute(fft_plan);
        fftw_destroy_plan(fft_plan);
        for (size_t i = 0; i < half_size; ++i) {
            double att = this->fresp[i];
            this->fftout[i][0] = this->fftout[i][0] * att;
            this->fftout[i][1] = this->fftout[i][1] * att;
        }

        memset(frame, 0, sizeof(double) * frame_size);
        fftw_plan ifft_plan = fftw_plan_dft_c2r_1d(frame_size, this->fftout, frame, FFTW_MEASURE);
        for (size_t i = 0; i < frame_size; ++i) {
            frame[i] /= static_cast<double>(frame_size);
        }

        fftw_execute(ifft_plan);
        fftw_destroy_plan(ifft_plan);

    }

};

class GeometricAttenuation
{
public:
    GeometricAttenuation()
    :sample_rate(0), max_delay(0), current_delay(nullptr), delayed_indexes(nullptr), indexes(nullptr)
    { }

    GeometricAttenuation(double fs, int channels)
    : sample_rate(fs)
    {
        this->current_delay = (double*) malloc(sizeof(double) * channels);
        memset(this->current_delay, 0, sizeof(double) * channels);
        this->max_delay = static_cast<size_t>(MAX_DELAY_SEC * this->sample_rate);
        this->delayed_indexes = nullptr;
        this->indexes = nullptr;
    }

    ~GeometricAttenuation() {
        if (this->current_delay) {
            free(this->current_delay);
            this->current_delay = nullptr;
        }
        free(this->indexes);
        free(this->delayed_indexes);
    }

    void apply_fractional_delay(double* frame, double* frame_out, double distance, int channel, size_t frame_size) {
        double delay = distance * this->sample_rate / SOUND_SPEED;
        double delta_delay = delay - this->current_delay[channel];
        if (delta_delay > SLEW_RATE) delta_delay = SLEW_RATE;
        if (delta_delay < -SLEW_RATE) delta_delay = -SLEW_RATE;
        this->current_delay[channel] += delta_delay;

        double* ind = (double*) realloc(this->indexes, sizeof(double) * frame_size);
        double* d_indexes = (double*) realloc(this->delayed_indexes, sizeof(double) * frame_size);
        if (!ind || !d_indexes) {
            std::cerr << "[ERROR] Failed realloc in fractional delay" << std::endl;
            return;
        }

        this->indexes = ind;
        this->delayed_indexes = d_indexes;

        for (size_t i = 0; i < frame_size; ++i) {
            this->indexes[i] = (int) i;
            this->delayed_indexes[i] = (double) i - this->current_delay[channel];
        }

        lerp(this->indexes, frame, this->delayed_indexes, frame_out, frame_size, frame_size, true);
    }

    double calculate_geometric_factor(double source_distance, double distance) {
        double original_distance = std::max(source_distance, ETA);
        return pow(original_distance / distance, GAMMA);
    }

private:
    double sample_rate;
    double max_delay;
    double* current_delay;
    double* delayed_indexes;
    double* indexes;
};

struct SpatialNeighs
{
    double a_dist;
    double b_dist;
    size_t i0;
    size_t i1;
};

SpatialNeighs spatial_match(CartesianPoint* target, HrirDatasetRead* dataset);
void cross_fade(double* prev_kernel, double* current_kernel, size_t transition_length);

#endif
