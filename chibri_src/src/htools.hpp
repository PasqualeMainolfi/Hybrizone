#ifndef HTOOLS_HPP
#define HTOOLS_HPP

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <fftw3.h>
#include <string>
#include <unordered_map>

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
#define CACHE_CAPACITY (4096)

inline double deg2rad(double deg_value) {
    return deg_value * M_PI / 180.0;
}

inline double rad2deg(double rad_value) {
    return 180.0 * rad_value / M_PI;
}

inline void unwrap_phase(double* x, size_t length) {
    double prev_left = x[0];
    double prev_right = x[1];

    for (size_t i = 1; i < length; ++i) {
        size_t li = i * 2;
        size_t ri = li + 1;

        double delta_left = x[li] - prev_left;
        double delta_right = x[ri] - prev_right;

        // Riporta delta nell'intervallo [-π, π]
        if (delta_left > M_PI) x[li] -= TWOPI;
        else if (delta_left < -M_PI) x[li] += TWOPI;

        if (delta_right > M_PI) x[ri] -= TWOPI;
        else if (delta_right < -M_PI) x[ri] += TWOPI;

        prev_left = x[li];
        prev_right = x[ri];
    }
}

enum ConvMode
{
    FULL,
    SAME
};

enum CacheType
{
    HRIR,
    RIR
};

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
        double norm = std::hypot(this->x, this->y, this->z);
        if (norm > 1e-10) {
            return CartesianPoint(this->x / norm, this->y / norm, this->z / norm);
        }
        return CartesianPoint(0.0, 0.0, 0.0);
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

    std::string get_polar_key() {
        return std::format("{:.5f}:{:.5f}:{:.5f}", this->rho, this->phi, this->theta);
    }

};

struct Hrir
{
    double* left_channel;
    double* right_channel;
    size_t channel_length;

    Hrir()
    : left_channel(nullptr), right_channel(nullptr), channel_length(0)
    { }

    Hrir(const Hrir& h)
    : channel_length(h.channel_length)
    {
        this->left_channel = (double*) malloc(sizeof(double) * this->channel_length);
        this->right_channel = (double*) malloc(sizeof(double) * this->channel_length);
        memcpy(this->left_channel, h.left_channel, sizeof(double) * this->channel_length);
        memcpy(this->right_channel, h.right_channel, sizeof(double) * this->channel_length);
    }

    Hrir(double* left, double* right, size_t channel_length)
    : channel_length(channel_length)
    {
        this->left_channel = (double*) malloc(sizeof(double) * this->channel_length);
        this->right_channel = (double*) malloc(sizeof(double) * this->channel_length);
        memcpy(this->left_channel, left, sizeof(double) * this->channel_length);
        memcpy(this->right_channel, right, sizeof(double) * this->channel_length);
    }

    ~Hrir() {
        if (this->left_channel) free(this->left_channel);
        if (this->right_channel) free(this->right_channel);
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

    }

    ~ISO9613Filter() {
        free(this->frequencies);
        free(this->fnorm);
        // free(this->fresp);
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

    // multiband filter (apply on a single channel Left and Right)
    void multiband_fft_filter(double* frame, double* alpha, size_t frame_size) {
        size_t half_size = frame_size / 2 + 1;
        double step = 1.0 / static_cast<double>(half_size - 1);

        double* ftemp = (double*) malloc(sizeof(double) * half_size);
        for (size_t i = 0; i < half_size; ++i) {
            ftemp[i] = step * (double) i;
        }

        double* fresp = (double*) malloc(sizeof(double) * half_size);
        lerp(this->fnorm, alpha, ftemp, fresp, NFREQS, half_size, false);
        free(ftemp);

        fftw_complex* temp_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
        fftw_plan fft_plan = fftw_plan_dft_r2c_1d(frame_size, frame, temp_fft, FFTW_MEASURE);
        fftw_execute(fft_plan);
        fftw_destroy_plan(fft_plan);

        for (size_t i = 0; i < half_size; ++i) {
            double att = fresp[i];
            temp_fft[i][0] *= att;
            temp_fft[i][1] *= att;
        }

        fftw_plan ifft_plan = fftw_plan_dft_c2r_1d(frame_size, temp_fft, frame, FFTW_ESTIMATE);

        fftw_execute(ifft_plan);
        fftw_destroy_plan(ifft_plan);

        for (size_t i = 0; i < frame_size; ++i) {
            frame[i] /= static_cast<double>(frame_size);
            // std::cout << frame[i] << std::endl;
        }

        fftw_free(temp_fft);
        free(fresp);
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
void fft_convolve(double** buffer, double* x, double* kernel, size_t x_size, size_t k_size, ConvMode conv_mode);

struct Morphdata
{
    fftw_complex* source_a;
    fftw_complex* source_b;
    fftw_complex* morphed;
    double smooth_factor;
    size_t lenght;
    size_t fftw_length;

    Morphdata()
    : source_a(nullptr), source_b(nullptr), morphed(nullptr), smooth_factor(0.0), lenght(0), fftw_length(0)
    { }

    Morphdata(const Morphdata& other) {
        this->smooth_factor = other.smooth_factor;
        this->lenght = other.lenght;
        this->fftw_length = other.fftw_length;

        this->source_a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->fftw_length);
        this->source_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->fftw_length);
        this->morphed = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->fftw_length);

        memcpy(this->source_a, other.source_a, sizeof(fftw_complex) * this->fftw_length);
        memcpy(this->source_b, other.source_b, sizeof(fftw_complex) * this->fftw_length);
        memcpy(this->morphed, other.morphed, sizeof(fftw_complex) * this->fftw_length);

    }

    ~Morphdata() {
        fftw_free(this->source_a);
        fftw_free(this->source_b);
        fftw_free(this->morphed);
    }
};

struct CacheNode
{
public:
    std::string key;
    void* value;
    CacheNode* prev_node;
    CacheNode* next_node;

    CacheNode(const std::string& key_, void* value_)
    : key(key_), value(value_), prev_node(nullptr), next_node(nullptr)
    { }
};

class LRUCache
{
public:
    size_t capacity;
    CacheType ctype;
    std::unordered_map<std::string, CacheNode*>* cache;
    CacheNode* head;
    CacheNode* tail;

    LRUCache(size_t cache_size, CacheType cache_type) {
        this->capacity = cache_size;
        this->ctype = cache_type;
        this->cache = new std::unordered_map<std::string, CacheNode*>();
        this->head = new CacheNode("HEAD", nullptr);
        this->tail = new CacheNode("TAIL", nullptr);
        this->head->next_node = this->tail;
        this->tail->prev_node = this->head;
    }

    ~LRUCache() {
        CacheNode* node = head;
        while (node != nullptr) {
            CacheNode* next = node->next_node;

            switch (this->ctype) {
                case CacheType::HRIR:
                    delete (Hrir*) node->value;
                    break;
                case CacheType::RIR:
                    delete (Morphdata*) node->value;
                    break;
            }

            delete node;
            node = next;
        }
        delete cache;
    }

    void put(const std::string& k, void* value) {
        if (this->cache->contains(k)) {
            this->move_to_head((*this->cache)[k]);
        } else {
            (*this->cache)[k] = new CacheNode(k, value);
            this->add((*this->cache)[k]);
        }

        if (this->cache->size() > this->capacity) {
            CacheNode* lru = this->tail->prev_node;
            this->remove(lru);

            switch (this->ctype) {
                case CacheType::HRIR:
                    delete (Hrir*) lru->value;
                    break;
                case CacheType::RIR:
                    delete (Morphdata*) lru->value;
                    break;
            }

            this->cache->erase(lru->key);
            delete lru;
        }
    }

    void* get(const std::string& k) {
        if (this->cache->contains(k)) {
           CacheNode* node = (*this->cache)[k];
           this->move_to_head(node);
           return node->value;
        }
        return nullptr;
    }

    bool contains(std::string& k) {
        return this->cache->contains(k);
    }


private:
    void remove(CacheNode* node) {
        CacheNode* prev = node->prev_node;
        CacheNode* next = node->next_node;
        prev->next_node = next;
        next->prev_node = prev;
    }

    void add(CacheNode* node) {
        node->next_node = this->head->next_node;
        node->prev_node = this->head;
        this->head->next_node->prev_node = node;
        this->head->next_node = node;
    }

    void move_to_head(CacheNode* node) {
        this->remove(node);
        this->add(node);
    }

};

enum CurveMode
{
    LINEAR,
    SIGMOID,
    EXPONENTIAL,
    LOGARITHMIC
};

struct RirFromDataset
{
    double* rir;
    size_t lenght;

    RirFromDataset()
    : rir(nullptr), lenght(0)
    { }

    ~RirFromDataset() {
        free(this->rir);
    }
};

struct Rir
{
    double* rir;
    size_t length;

    Rir()
    : rir(nullptr), length(0)
    { }

    Rir(const Rir& r) {
        this->rir = (double*) malloc(sizeof(double) * length);
        memcpy(this->rir, r.rir, sizeof(double) * length);
        this->length = r.length;
    }

    Rir(double* buffer, size_t channel_length) {
        this->rir = (double*) malloc(sizeof(double) * channel_length);
        memcpy(this->rir, buffer, sizeof(double) * channel_length);
        this->length = channel_length;
    }

    ~Rir() {
        if (this->rir) free(this->rir);
    }
};


class RirDatasetRead
{
public:
    H5::H5File rdata;
    char** ir_keys;

    RirDatasetRead() = default;
    RirDatasetRead(const char* dataset_path) {
        this->rdata = H5::H5File(dataset_path, H5F_ACC_RDONLY);

        H5::Attribute keys = this->rdata.openAttribute("IR-keys");
        H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
        string_type.setCset(H5T_CSET_UTF8);
        string_type.setStrpad(H5T_STR_NULLTERM);

        H5::DataSpace space = keys.getSpace();
        hsize_t dim[1];
        space.getSimpleExtentDims(dim);
        this->ir_keys = new char*[dim[0]];
        keys.read(string_type, this->ir_keys);
        this->data_length = dim[0];

        H5::Attribute fs = this->rdata.openAttribute("fs");
        fs.read(H5::PredType::NATIVE_INT64, &this->sample_rate);

        this->temp_rir = nullptr;
    }

    ~RirDatasetRead() {
        for (size_t i = 0; i < this->data_length; ++i) {
            free(this->ir_keys[i]);
        }
        delete[] this->ir_keys;
        if (this->temp_rir) free(this->temp_rir);
    }

    void get_rdata(RirFromDataset* rir_buffer, size_t index) {
        const char* key = this->ir_keys[index];
        H5::DataSet r = this->rdata.openDataSet(key);
        H5::DataSpace space = r.getSpace();
        hsize_t dim[1];
        space.getSimpleExtentDims(dim);

        double* temp = (double*) realloc(this->temp_rir, sizeof(double) * dim[0]);
        this->temp_rir = temp;
        r.read(this->temp_rir, H5::PredType::NATIVE_DOUBLE);

        rir_buffer->rir = (double*) malloc(sizeof(double) * dim[0]);
        rir_buffer->lenght = dim[0];
        memcpy(rir_buffer->rir, this->temp_rir, sizeof(double) * dim[0]);
    }

    double get_sample_rate() {
        return (double) this->sample_rate;
    }

private:
    size_t data_length;
    int64_t sample_rate;
    double* temp_rir;
};

#endif
