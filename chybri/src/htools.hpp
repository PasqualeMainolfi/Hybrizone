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
#include <fftw3.h>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#define P_REF (101325.0)
#define T_ZERO (293.15)
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
#define OSA_TRANSITION_FACTOR (0.5)
#define MAX_TRANSITION_SAMPLES (512)
#define SOFT_CLIP_FACTOR (1.0 / 0.707)
#define MAX_OSA_BUFFER_SIZE (22050)
#define EPSILON (1e-9)


inline void from_interleaved_to_single(double* interleaved, double* a, double* b, size_t out_size) {
    for (size_t i = 0; i < out_size; ++i) {
        a[i] = interleaved[i * 2];
        b[i] = interleaved[i * 2 + 1];
    }
}

inline void from_fftw_complex_to_double_interleaved(fftw_complex* x, double* interleaved, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        interleaved[i * 2] = x[i][0];
        interleaved[i * 2 + 1] = x[i][1];
    }
}

inline size_t next_power_of_two(size_t n) {
    if (n == 0) return 1;
    return static_cast<size_t>(pow(2, std::ceil(std::log2(n))));
}

inline double deg2rad(double deg_value) {
    return deg_value * M_PI / 180.0;
}

inline double rad2deg(double rad_value) {
    return 180.0 * rad_value / M_PI;
}

inline void unwrap_phase(double* x, size_t half_size) {
    // left channel
    double left_prev = x[0];
    double offset_l = 0.0;
    for (size_t i = 1; i < half_size; ++i) {
        size_t current_idx = i * 2;
        double diff = x[current_idx] - left_prev;
        left_prev = x[current_idx];

        if (diff > M_PI) {
            offset_l -= TWOPI;
        } else if (diff < -M_PI) {
            offset_l += TWOPI;
        }

        x[current_idx] += offset_l;
    }

    // right channel
    double right_prev = x[1];
    double offset_r = 0.0;
    for (size_t i = 1; i < half_size; ++i) {
        size_t current_idx = i * 2 + 1;
        double diff = x[current_idx] - right_prev;
        right_prev = x[current_idx];

        if (diff > M_PI) {
            offset_r -= TWOPI;
        } else if (diff < -M_PI) {
            offset_r += TWOPI;
        }

        x[current_idx] += offset_r;
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
    HANGLE
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
        if (norm > EPSILON) {
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
        return std::format("{:.5f}_{:.5f}_{:.5f}", this->rho, this->phi, this->theta);
    }

};

struct SlerpCoeff
{
    double a;
    double b;
};

inline SlerpCoeff slerp_coefficients(CartesianPoint* p, CartesianPoint* p1, CartesianPoint* p2) {
    CartesianPoint trgnorm = p->normalize();
    CartesianPoint h1norm = p1->normalize();
    CartesianPoint h2norm = p2->normalize();

    double dot = h1norm.x * h2norm.x + h1norm.y * h2norm.y + h1norm.z * h2norm.z;
    dot = std::min(std::max(dot, -1.0), 1.0);
    double omega = std::acos(dot);

    double a, b;
    if (omega == 0.0) {
        a = 0.0;
        b = 1.0;
    } else {
        double tdot1 = trgnorm.x * h1norm.x + trgnorm.y * h1norm.y + trgnorm.z * h1norm.z;
        tdot1 = std::min(std::max(tdot1, -1.0), 1.0);
        double omega1 = acos(tdot1);
        double tdot2 = trgnorm.x * h2norm.x + trgnorm.y * h2norm.y + trgnorm.z * h2norm.z;
        tdot2 = std::min(std::max(tdot2, -1.0), 1.0);
        double omega2 = acos(tdot2);
        double sin_omega = sin(omega);
        double alpha = omega1 / (omega1 + omega2);
        a = sin((1.0 - alpha) * omega) / sin_omega;
        b = sin(alpha * omega) / sin_omega;
    }

    return SlerpCoeff { .a = a, .b = b};
};

struct Hrir
{
    std::vector<double> left_channel;
    std::vector<double> right_channel;
    size_t channel_length;

    Hrir()
    : left_channel(std::vector<double>()), right_channel(std::vector<double>()), channel_length(0)
    { }

    Hrir(const Hrir& h)
    : channel_length(h.channel_length)
    {
        this->left_channel.resize(this->channel_length);
        this->right_channel.resize(this->channel_length);
        memcpy(this->left_channel.data(), h.left_channel.data(), sizeof(double) * this->channel_length);
        memcpy(this->right_channel.data(), h.right_channel.data(), sizeof(double) * this->channel_length);
    }

    Hrir(double* left, double* right, size_t channel_length)
    : channel_length(channel_length)
    {
        this->left_channel.resize(this->channel_length);
        this->right_channel.resize(this->channel_length);
        memcpy(this->left_channel.data(), left, sizeof(double) * this->channel_length);
        memcpy(this->right_channel.data(), right, sizeof(double) * this->channel_length);
    }

    ~Hrir() { }
};

class HrirDatasetRead
{
    H5::H5File hdata;
    int* polar_index;
    double* regular_coords;
    HShape hrir_shape;
    double source_distance;
    double sample_rate;
    size_t dataset_size;

public:
    HrirDatasetRead(const char* dataset_path) {
        this->hdata = H5::H5File(dataset_path, H5F_ACC_RDONLY);

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
    void get_hrir_data(std::vector<double>* buffer, const char* key, HDataType data_type) {
        H5::DataSet h;
        std::string k;
        hsize_t dims[2];
        switch (data_type) {
            case HDataType::HTIME:
                k = std::format("/hrir/{}", key);
                h = this->hdata.openDataSet(k);
                break;
            case HDataType::HMAG:
                k = std::format("/hrir_fft/{}/mag", key);
                h = this->hdata.openDataSet(k);
                break;
            case HDataType::HANGLE:
                k = std::format("/hrir_fft/{}/angle", key);
                h = this->hdata.openDataSet(k);
                break;
        }

        H5::DataSpace dataspace = h.getSpace();
        dataspace.getSimpleExtentDims(dims);
        size_t buffer_size = dims[0] * dims[1];
        buffer->resize(buffer_size);
        h.read(buffer->data(), H5::PredType::NATIVE_DOUBLE);
    }

    void get_itd(double* buffer, const char* key) {
        std::string k = std::format("hrir_itd/{}", key);
        H5::DataSet h = this->hdata.openDataSet(k);
        h.read(buffer, H5::PredType::NATIVE_DOUBLE);
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
        this->frequencies = std::vector<double>(NFREQS, 0.0);
        this->fnorm = std::vector<double>(NFREQS, 0.0);

        double nyq = this->sample_rate / 2.0;
        double fstep = nyq / static_cast<double>(NFREQS - 1);
        for (size_t i = 0; i < NFREQS; ++i) {
            double value = i * fstep;
            this->frequencies[i] = value;
            this->fnorm[i] = value / nyq;
        }

    }

    ~ISO9613Filter() = default;

    void get_attenuation_air_absorption(double* alpha) {
        double p_sat = P_REF * (pow(10.0, (-6.8346 * pow((273.16 / this->air_data->kelvin), 1.261) + 4.6151)));
        double h = this->air_data->rh * (p_sat / (this->air_data->p_atm * P_REF));
        double tr = this->air_data->kelvin / T_ZERO;
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

        std::vector<double> new_alpha(NFREQS);
        for (size_t i = 0; i < NFREQS; ++i) {
            double db_attenuation = alpha[i] * rdist;
            new_alpha[i] = pow(10.0, -db_attenuation / 20.0);
        }

        new_alpha[NFREQS - 1] = 0.0;
        this->multiband_fft_filter(frame, new_alpha.data(), frame_size);
    }

private:
    double sample_rate;
    std::vector<double> frequencies;
    std::vector<double> fnorm;

    // multiband filter (apply on a single channel Left and Right)
    void multiband_fft_filter(double* frame, double* alpha, size_t frame_size) {
        size_t half_size = frame_size / 2 + 1;
        double step = 1.0 / static_cast<double>(half_size - 1);

        std::vector<double> ftemp(half_size, 0.0);
        for (size_t i = 0; i < half_size; ++i) {
            ftemp[i] = step * (double) i;
        }

        std::vector<double> fresp(half_size, 0.0);
        lerp(this->fnorm.data(), alpha, ftemp.data(), fresp.data(), NFREQS, half_size, false);

        fftw_complex* temp_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_size);
        fftw_plan fft_plan = fftw_plan_dft_r2c_1d(frame_size, frame, temp_fft, FFTW_ESTIMATE);
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
        }

        fftw_free(temp_fft);
    }

};

class GeometricAttenuation
{
public:
    GeometricAttenuation()
    :sample_rate(0), max_delay(0), current_delay(std::vector<double>()), delayed_indexes(std::vector<double>()), indexes(std::vector<double>())
    { }

    GeometricAttenuation(double fs, int channels)
    : sample_rate(fs)
    {
        this->current_delay = std::vector<double>(channels, 0);
        this->max_delay = static_cast<size_t>(MAX_DELAY_SEC * this->sample_rate);
        this->delayed_indexes = std::vector<double>();
        this->indexes = std::vector<double>();
    }

    ~GeometricAttenuation() = default;

    void alloc_indexes(size_t length) {
        this->indexes.resize(length);
        std::iota(this->indexes.begin(), this->indexes.end(), 0);
        this->delayed_indexes.resize(length);
    }

    void apply_fractional_delay(double* frame, double* frame_out, double distance, int channel, size_t frame_size) {
        double delay = distance * this->sample_rate / SOUND_SPEED;
        double delta_delay = delay - this->current_delay[channel];
        delta_delay = std::max(-SLEW_RATE, std::min(SLEW_RATE, delta_delay));
        this->current_delay[channel] += delta_delay;

        for (size_t i = 0; i < frame_size; ++i) {
            this->indexes[i] = static_cast<double>(i);
            this->delayed_indexes[i] = this->indexes[i] - this->current_delay[channel];
        }

        lerp(this->indexes.data(), frame, this->delayed_indexes.data(), frame_out, frame_size, frame_size, true);
    }

    double calculate_geometric_factor(double source_distance, double distance) {
        double original_distance = std::max(source_distance, ETA);
        return pow(original_distance / distance, GAMMA);
    }

private:
    double sample_rate;
    double max_delay;
    std::vector<double> current_delay;
    std::vector<double> delayed_indexes;
    std::vector<double> indexes;
};

struct SpatialNeighs
{
    double a_dist;
    double b_dist;
    size_t i0;
    size_t i1;
};

SpatialNeighs spatial_match(CartesianPoint* target, HrirDatasetRead* dataset);
void cross_fade(double* prev_kernel, double* current_kernel, double* a, double* b, size_t transition_length);
void fft_convolve(std::vector<double>* buffer, double* x, double* kernel, size_t x_size, size_t k_size, ConvMode conv_mode);

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
    std::vector<double> rir;
    size_t lenght;

    RirFromDataset()
    : rir(std::vector<double>()), lenght(0)
    { }

    ~RirFromDataset() = default;
};

struct Rir
{
    std::vector<double> rir;
    size_t length;

    Rir()
    : rir(std::vector<double>()), length(0)
    { }

    Rir(const Rir& r) {
        this->rir.resize(r.length);
        memcpy(this->rir.data(), r.rir.data(), sizeof(double) * r.length);
        this->length = r.length;
    }

    Rir(double* buffer, size_t channel_length) {
        this->rir.resize(channel_length);
        memcpy(this->rir.data(), buffer, sizeof(double) * channel_length);
        this->length = channel_length;
    }

    ~Rir() = default;
};


class RirDatasetRead
{
public:
    H5::H5File rdata;
    char** ir_keys;

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

        rir_buffer->rir.resize(dim[0]);
        rir_buffer->lenght = dim[0];
        memcpy(rir_buffer->rir.data(), this->temp_rir, sizeof(double) * dim[0]);
    }

    double get_sample_rate() {
        return static_cast<double>(this->sample_rate);
    }

private:
    size_t data_length;
    int64_t sample_rate;
    double* temp_rir;
};

struct OSABuffer
{
    std::vector<double> buffer;
    size_t conv_buffer_size;

    OSABuffer()
    : buffer(std::vector<double>()), conv_buffer_size(0)
    { }

    ~OSABuffer() = default;
};

void apply_intermediate(OSABuffer* osa_buffer, double* x, std::vector<double> prev_kernel, std::vector<double> curr_kernel, size_t x_length, size_t prev_kernel_length, size_t curr_kernel_length, size_t transition_length);
void intermediate_segment(double* buffer, double* x, double* prev_kernel, double* curr_kernel, size_t ksize, size_t transition_size);

class OSAConv
{
public:
    OSAConv(size_t chunk_size)
    : buffer_size(chunk_size)
    {
        this->init_buffer_size = MAX_OSA_BUFFER_SIZE;
        this->buffer = std::vector<double>(this->init_buffer_size, 0.0);
        this->transition_size = this->buffer_size < 1024 ? static_cast<size_t>(this->buffer_size * OSA_TRANSITION_FACTOR) : MAX_TRANSITION_SAMPLES;
        this->prev_kernel = std::vector<double>();
        this->prev_kernel_size = 0;
    }

    ~OSAConv() = default;

    void process(double* buffer_out, double* x, std::vector<double> kernel, size_t kernel_size) {

        OSABuffer b;
        apply_intermediate(&b, x, this->prev_kernel, kernel, this->buffer_size, this->prev_kernel_size, kernel_size, this->transition_size);

        this->prev_kernel = kernel;
        this->prev_kernel_size = kernel_size;

        for (size_t i = 0; i < this->buffer_size; ++i) {
            b.buffer[i] += this->buffer[i];
        }

        size_t shift_size = this->init_buffer_size - this->buffer_size;
        memmove(this->buffer.data(), this->buffer.data() + this->buffer_size, sizeof(double) * shift_size);
        memset(this->buffer.data() + shift_size, 0, sizeof(double) * this->buffer_size);

        memcpy(buffer_out, b.buffer.data(), sizeof(double) * this->buffer_size); // frame out buffer

        int tail_size = static_cast<int>(b.conv_buffer_size) - static_cast<int>(this->buffer_size);
        if (tail_size > 0) {
            if (static_cast<double>(tail_size) > this->init_buffer_size) {
                std::vector<double> temp_internal_buffer(tail_size, 0.0);
                memcpy(temp_internal_buffer.data(), this->buffer.data(), sizeof(double) * this->init_buffer_size);
                this->buffer = temp_internal_buffer;
                this->init_buffer_size = tail_size;
            }

            for (int i = 0; i < tail_size; ++i) {
                this->buffer[i] += b.buffer[i + this->buffer_size];
            }
        }
    }

private:
    size_t init_buffer_size;
    size_t buffer_size;
    size_t transition_size;
    std::vector<double> buffer;
    std::vector<double> prev_kernel;
    size_t prev_kernel_size;
};

struct HybriOuts
{
    std::vector<double> left_channel;
    std::vector<double> right_channel;
    size_t buffer_size;

    HybriOuts()
    : left_channel(std::vector<double>()), right_channel(std::vector<double>()), buffer_size(0)
    { }

    ~HybriOuts() = default;

    std::vector<float> get_float_interleaved() {
        std::vector<float> interleaved(this->buffer_size * 2);
        for (size_t i = 0; i < this->buffer_size; ++i) {
            interleaved[i * 2] = static_cast<float>(this->left_channel[i]);
            interleaved[i * 2 + 1] = static_cast<float>(this->right_channel[i]);
        }
        return interleaved;
    }
};

#endif
