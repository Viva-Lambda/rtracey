// vec3.hpp for cuda
#ifndef VEC3_CUH
#define VEC3_CUH

#include <external.hpp>
#include <utils.cuh>

class Vec3 {
public:
  float e[3];

  __host__ __device__ Vec3() {}
  __host__ __device__ Vec3(float e1, float e2, float e3) {
    e[0] = e1;
    e[1] = e2;
    e[2] = e3;
  }
  __host__ __device__ Vec3(float e1) {
    e[0] = e1;
    e[1] = e1;
    e[2] = e1;
  }
  __host__ __device__ Vec3(float es[3]) {
    e[0] = es[0];
    e[1] = es[1];
    e[2] = e[2];
  }
  __host__ __device__ inline float x() const {
    return e[0];
  }
  __host__ __device__ inline float y() const {
    return e[1];
  }
  __host__ __device__ inline float z() const {
    return e[2];
  }
  __host__ __device__ inline float r() const { return x(); }
  __host__ __device__ inline float g() const { return y(); }
  __host__ __device__ inline float b() const { return z(); }

  __host__ __device__ inline Vec3 operator-() const {
    return Vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ inline float operator[](int i) const {
    return e[i];
  }
  __host__ __device__ inline float &operator[](int i) {
    return e[i];
  }
  __host__ __device__ inline Vec3 &
  operator+=(const Vec3 &v);
  __host__ __device__ inline Vec3 &
  operator-=(const Vec3 &v);
  __host__ __device__ inline Vec3 &
  operator*=(const Vec3 &v);
  __host__ __device__ inline Vec3 &
  operator/=(const Vec3 &v);
  __host__ __device__ inline Vec3 &
  operator*=(const float t);
  __host__ __device__ inline Vec3 &
  operator/=(const float t);
  __host__ __device__ inline Vec3 &
  operator+=(const float t);
  __host__ __device__ inline Vec3 &
  operator-=(const float t);

  __host__ __device__ inline float squared_length() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
  __host__ __device__ inline float length() const {
    return sqrt(squared_length());
  }
  __host__ __device__ inline Vec3 to_unit() const;
  __host__ __device__ inline void unit_vector() const;
  __host__ std::vector<float> to_v() const {
    std::vector<float> v(3);
    v[0] = x();
    v[1] = y();
    v[2] = z();
    return v;
  }
  __host__ __device__ inline static Vec3
  random(unsigned int seed) {
    return Vec3(randf(seed), randf(seed), randf(seed));
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const Vec3 &t) {
  os << t.x() << " " << t.y() << " " << t.z();
  return os;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1,
                                          const Vec3 &v2) {
  return Vec3(v1.x() + v2.x(), v1.y() + v2.y(),
              v1.z() + v2.z());
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1,
                                          const Vec3 &v2) {
  return Vec3(v1.x() - v2.x(), v1.y() - v2.y(),
              v1.z() - v2.z());
}
__host__ __device__ inline Vec3 operator*(const Vec3 &v1,
                                          const Vec3 &v2) {
  return Vec3(v1.x() * v2.x(), v1.y() * v2.y(),
              v1.z() * v2.z());
}
__host__ __device__ inline Vec3 operator/(const Vec3 &v1,
                                          const Vec3 &v2) {
  return Vec3(v1.x() / v2.x(), v1.y() / v2.y(),
              v1.z() / v2.z());
}
__host__ __device__ inline Vec3 operator*(const Vec3 &v1,
                                          float t) {
  return Vec3(v1.x() * t, v1.y() * t, v1.z() * t);
}
__host__ __device__ inline Vec3 operator/(const Vec3 &v1,
                                          float t) {
  return Vec3(v1.x() / t, v1.y() / t, v1.z() / t);
}
__host__ __device__ inline Vec3 operator+(const Vec3 &v1,
                                          float t) {
  return Vec3(v1.x() + t, v1.y() + t, v1.z() + t);
}
__host__ __device__ inline Vec3 operator-(const Vec3 &v1,
                                          float t) {
  return Vec3(v1.x() - t, v1.y() - t, v1.z() - t);
}

__host__ __device__ inline float dot(const Vec3 &v1,
                                     const Vec3 &v2) {
  return v1.x() * v2.x() + v1.y() * v2.y() +
         v1.z() * v2.z();
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1,
                                      const Vec3 &v2) {
  return Vec3((v1.y() * v2.z() - v1.z() * v2.y()),
              (-(v1.x() * v2.z() - v1.z() * v2.x())),
              (v1.x() * v2.y() - v1.y() * v2.x()));
}

__host__ __device__ inline Vec3 &Vec3::
operator+=(const Vec3 &v) {
  e[0] += v.x();
  e[1] += v.y();
  e[2] += v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator*=(const Vec3 &v) {
  e[0] *= v.x();
  e[1] *= v.y();
  e[2] *= v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator/=(const Vec3 &v) {
  e[0] /= v.x();
  e[1] /= v.y();
  e[2] /= v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator-=(const Vec3 &v) {
  e[0] -= v.x();
  e[1] -= v.y();
  e[2] -= v.z();
  return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator+=(float v) {
  e[0] += v;
  e[1] += v;
  e[2] += v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator-=(float v) {
  e[0] -= v;
  e[1] -= v;
  e[2] -= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator*=(float v) {
  e[0] *= v;
  e[1] *= v;
  e[2] *= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator/=(float v) {
  e[0] /= v;
  e[1] /= v;
  e[2] /= v;
  return *this;
}
__host__ __device__ inline Vec3 to_unit(Vec3 v) {
  return v / v.length();
}
__host__ __device__ inline float distance(Vec3 v1,
                                          Vec3 v2) {
  return (v1 - v2).length();
}

__host__ __device__ Vec3 min_vec(const Vec3 &v1,
                                 const Vec3 &v2) {
  float xmin = fmin(v1.x(), v2.x());
  float ymin = fmin(v1.y(), v2.y());
  float zmin = fmin(v1.z(), v2.z());
  return Vec3(xmin, ymin, zmin);
}
__host__ __device__ Vec3 max_vec(const Vec3 v1,
                                 const Vec3 v2) {
  float xmax = fmax(v1.x(), v2.x());
  float ymax = fmax(v1.y(), v2.y());
  float zmax = fmax(v1.z(), v2.z());
  return Vec3(xmax, ymax, zmax);
}
__host__ __device__ Vec3 sqrt(const Vec3 &v1) {
  float x = sqrt(v1.x());
  float y = sqrt(v1.y());
  float z = sqrt(v1.z());
  return Vec3(x, y, z);
}

__host__ __device__ bool refract(const Vec3 &v,
                                 const Vec3 &n,
                                 float ni_over_nt,
                                 Vec3 &refracted) {
  Vec3 uv = to_unit(v);
  float dt = dot(uv, n);
  float discriminant =
      1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
  if (discriminant > 0) {
    refracted =
        ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  } else {
    return false;
  }
}

__host__ __device__ Vec3 reflect(const Vec3 &v,
                                 const Vec3 &n) {
  return v - 2.0f * dot(v, n) * n;
}

#define RND (curand_uniform(&local_rand_state))

__device__ float random_float(curandState *loc, float min,
                              float max) {
  return min + (max - min) * curand_uniform(loc);
}
__device__ int random_int(curandState *loc) {
  return (int)curand_uniform(loc);
}
__device__ int random_int(curandState *loc, int mn,
                          int mx) {
  return (int)random_float(loc, (float)mn, (float)mx);
}

__host__ float h_random_float(float min, float max) {
  return min + (max - min) * hrandf();
}
__host__ int h_random_int(int mn, int mx) {
  return (int)h_random_float((float)mn, (float)mx);
}

__host__ __device__ float rand_float(unsigned int seed) {
  return randf(seed);
}

__device__ Vec3 random_vec(curandState *local_rand_state) {
  return Vec3(curand_uniform(local_rand_state),
              curand_uniform(local_rand_state),
              curand_uniform(local_rand_state));
}
__host__ Vec3 h_random_vec() {
  return Vec3(hrandf(), hrandf(), hrandf());
}
__device__ Vec3 random_vec(curandState *local_rand_state,
                           float mn, float mx) {
  return Vec3(random_float(local_rand_state, mn, mx),
              random_float(local_rand_state, mn, mx),
              random_float(local_rand_state, mn, mx));
}
__host__ Vec3 h_random_vec(float mn, float mx) {
  return Vec3(h_random_float(mn, mx),
              h_random_float(mn, mx),
              h_random_float(mn, mx));
}
__device__ Vec3 random_vec(curandState *local_rand_state,
                           float mx, float my, float mz) {
  return Vec3(random_float(local_rand_state, 0, mx),
              random_float(local_rand_state, 0, my),
              random_float(local_rand_state, 0, mz));
}
__host__ Vec3 h_random_vec(float mx, float my, float mz) {
  return Vec3(h_random_float(0, mx), h_random_float(0, my),
              h_random_float(0, mz));
}

__device__ Vec3
random_in_unit_sphere(curandState *local_rand_state) {
  while (true) {
    Vec3 p =
        2.0f * random_vec(local_rand_state, -1.0f, 1.0f) -
        Vec3(1.0f);
    if (p.squared_length() < 1.0f)
      return p;
  }
}
__host__ Vec3 h_random_in_unit_sphere() {
  while (true) {
    Vec3 p = 2.0f * h_random_vec(-1.0f, 1.0f) - Vec3(1.0f);
    if (p.squared_length() < 1.0f)
      return p;
  }
}
__device__ Vec3 random_in_unit_disk(curandState *lo) {
  while (true) {
    Vec3 p = 2.0 * Vec3(random_float(lo, -1.0f, 1.0f),
                        random_float(lo, -1.0f, 1.0f), 0) -
             Vec3(1, 1, 0);
    if (p.squared_length() < 1.0f)
      return p;
  }
}
__host__ Vec3 h_random_in_unit_disk() {
  while (true) {
    Vec3 p = 2.0 * Vec3(h_random_float(-1.0f, 1.0f),
                        h_random_float(-1.0f, 1.0f), 0) -
             Vec3(1, 1, 0);
    if (p.squared_length() < 1.0f)
      return p;
  }
}

__device__ Vec3 random_in_hemisphere(curandState *lo,
                                     Vec3 normal) {
  Vec3 in_unit_sphere = random_in_unit_sphere(lo);
  if (dot(in_unit_sphere, normal) > 0.0)
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}
__host__ Vec3 h_random_in_hemisphere(Vec3 normal) {
  Vec3 in_unit_sphere = h_random_in_unit_sphere();
  if (dot(in_unit_sphere, normal) > 0.0)
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

__device__ Vec3 random_cosine_direction(curandState *lo) {
  auto r1 = curand_uniform(lo);
  auto r2 = curand_uniform(lo);
  auto z = sqrt(1 - r2);

  auto phi = 2 * M_PI * r1;
  auto x = cos(phi) * sqrt(r2);
  auto y = sin(phi) * sqrt(r2);

  return Vec3(x, y, z);
}
__host__ Vec3 h_random_cosine_direction() {
  auto r1 = hrandf();
  auto r2 = hrandf();
  auto z = sqrt(1 - r2);

  auto phi = 2 * M_PI * r1;
  auto x = cos(phi) * sqrt(r2);
  auto y = sin(phi) * sqrt(r2);

  return Vec3(x, y, z);
}
__device__ inline Vec3
random_to_sphere(float radius, float distance_squared,
                 curandState *loc) {
  auto r1 = curand_uniform(loc);
  auto r2 = curand_uniform(loc);
  auto z =
      1 +
      r2 * (sqrt(1 - radius * radius / distance_squared) -
            1);

  auto phi = 2 * M_PI * r1;
  auto x = cos(phi) * sqrt(1 - z * z);
  auto y = sin(phi) * sqrt(1 - z * z);

  return Vec3(x, y, z);
}
__host__ inline Vec3
h_random_to_sphere(float radius, float distance_squared) {
  auto r1 = hrandf();
  auto r2 = hrandf();
  auto z =
      1 +
      r2 * (sqrt(1 - radius * radius / distance_squared) -
            1);

  auto phi = 2 * M_PI * r1;
  auto x = cos(phi) * sqrt(1 - z * z);
  auto y = sin(phi) * sqrt(1 - z * z);

  return Vec3(x, y, z);
}

using Point3 = Vec3;
using Color = Vec3;

#endif
