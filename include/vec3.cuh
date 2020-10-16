#pragma once
#include <external.hpp>
#include <utils.cuh>

class Vec3 {
public:
  float *e1;
  float *e2;
  float *e3;

  __host__ __device__ Vec3()
      : e1(nullptr), e2(nullptr), e3(nullptr) {}
  __host__ __device__ Vec3(float _e1, float _e2,
                           float _e3) {
    e1 = &_e1;
    e2 = &_e2;
    e3 = &_e3;
  }
  __host__ __device__ Vec3(float *_e1, float *_e2,
                           float *_e3) {
    e1 = _e1;
    e2 = _e2;
    e3 = _e3;
  }

  __host__ __device__ Vec3(float es[3]) {
    e1 = &es[0];
    e2 = &es[1];
    e3 = &es[2];
  }
  __host__ __device__ Vec3(const Vec3 &v) {
    e1 = v.e1;
    e2 = v.e2;
    e3 = v.e3;
  }
  __host__ __device__ inline float x() const { return *e1; }
  __host__ __device__ inline float y() const { return *e2; }
  __host__ __device__ inline float z() const { return *e3; }
  __host__ __device__ inline float r() const { return x(); }
  __host__ __device__ inline float g() const { return y(); }
  __host__ __device__ inline float b() const { return z(); }

  __host__ __device__ inline Vec3 operator-() const {
    return Vec3(-x(), -y(), -z());
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
  __host__ __device__ inline float operator[](int i) const {
    if (i <= 0) {
      return x();
    } else if (i == 1) {
      return y();
    } else {
      return z();
    }
  }
  __host__ __device__ inline float &operator[](int i) {
    if (i <= 0) {
      return *e1;
    } else if (i == 1) {
      return *e2;
    } else {
      return *e3;
    }
  }
  __host__ __device__ inline float squared_length() const {
    return x() * x() + y() * y() + z() * z();
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
__host__ __device__ inline Vec3 operator*(float t,
                                          const Vec3 &v1) {
  return v1 * t;
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
__host__ __device__ inline Vec3 operator-(float t,
                                          const Vec3 &v1) {
  return Vec3(t - v1.x(), t - v1.y(), t - v1.z());
}

__host__ __device__ inline float dot(const Vec3 &v1,
                                     const Vec3 &v2) {
  return v1.x() * v2.x() + v1.y() * v2.y() +
         v1.z() * v2.z();
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1,
                                      const Vec3 &v2) {
  return Vec3(v1.y() * v2.z() - v1.z() * v2.y(),
              v1.z() * v2.x() - v1.x() * v2.z(),
              (v1.x() * v2.y() - v1.y() * v2.x()));
}

__host__ __device__ inline Vec3 &Vec3::
operator+=(const Vec3 &v) {
  *e1 += v.x();
  *e2 += v.y();
  *e3 += v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator*=(const Vec3 &v) {
  *e1 *= v.x();
  *e2 *= v.y();
  *e3 *= v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator/=(const Vec3 &v) {
  *e1 /= v.x();
  *e2 /= v.y();
  *e3 /= v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator-=(const Vec3 &v) {
  *e1 -= v.x();
  *e2 -= v.y();
  *e3 -= v.z();
  return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator+=(float v) {
  *e1 += v;
  *e2 += v;
  *e3 += v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator-=(float v) {
  *e1 -= v;
  *e2 -= v;
  *e3 -= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator*=(float v) {
  *e1 *= v;
  *e2 *= v;
  *e3 *= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator/=(float v) {
  *e1 /= v;
  *e2 /= v;
  *e3 /= v;
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

__host__ __device__ bool refract(const Vec3 &v,
                                 const Vec3 &n,
                                 float ni_over_nt,
                                 Vec3 &refracted) {
  Vec3 uv = to_unit(v);
  float dt = dot(uv, n);
  float discriminant =
      1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
  if (discriminant > 0) {
    Vec3 uv_n = uv - n;
    refracted =
        ni_over_nt * (uv_n * dt) - n * sqrt(discriminant);
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

__device__ Vec3 random_vec(curandState *loc) {
  return Vec3(curand_uniform(loc), curand_uniform(loc),
              curand_uniform(loc));
}
__device__ Vec3 random_vec(curandState *loc, float mn,
                           float mx) {
  return Vec3(random_float(loc, mn, mx),
              random_float(loc, mn, mx),
              random_float(loc, mn, mx));
}
__device__ Vec3 random_vec(curandState *local_rand_state,
                           float mx, float my, float mz) {
  return Vec3(random_float(local_rand_state, 0, mx),
              random_float(local_rand_state, 0, my),
              random_float(local_rand_state, 0, mz));
}

__device__ Vec3
random_in_unit_sphere(curandState *local_rand_state) {
  while (true) {
    Vec3 p =
        2.0f * random_vec(local_rand_state, -1.0f, 1.0f) -
        Vec3(1.0f, 1.0f, 1.0f);
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

__device__ Vec3 random_in_hemisphere(curandState *lo,
                                     Vec3 normal) {
  Vec3 in_unit_sphere = random_in_unit_sphere(lo);
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

using Point3 = Vec3;
using Color = Vec3;
