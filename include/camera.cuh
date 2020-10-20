#pragma once

#include <ray.cuh>
#include <vec3.cuh>

/** A simple Camera specification object that can be
  instantiated both
  on host and on device.
 */
class Camera {
public:
  __host__ __device__ Camera(Vec3 orig, Vec3 target,
                             Vec3 vup, float vfov,
                             float aspect, float aperture,
                             float focus_dist,
                             float t0 = 0.0f,
                             float t1 = 0.0f) {
    lens_radius = aperture / 2;
    time0 = t0;
    time1 = t1;
    float theta = vfov * M_PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;
    origin = orig;
    w = to_unit(orig - target);
    u = to_unit(cross(vup, w));
    v = cross(w, u);
    lower_left_corner =
        origin - half_width * focus_dist * u -
        half_height * focus_dist * v - focus_dist * w;
    horizontal = 2 * half_width * focus_dist * u;
    vertical = 2 * half_height * focus_dist * v;
  }

  __host__ __device__ Ray _get_ray(float s, float t,
                                   Vec3 riud,
                                   float rf) const {
    //
    Vec3 rd = lens_radius * riud;
    Vec3 offset = u * rd.x() + v * rd.y();
    return Ray(origin + offset,
               lower_left_corner + s * horizontal +
                   t * vertical - origin - offset,
               rf);
  }

  __device__ Ray get_ray(float s, float t,
                         curandState *lo) const {
    Vec3 riud = random_in_unit_disk(lo);
    float rf = random_float(lo, time0, time1);
    return _get_ray(s, t, riud, rf);
  }
  __host__ Ray h_get_ray(float s, float t) const {
    Vec3 riud = h_random_in_unit_disk();
    float rf = h_random_float(time0, time1);
    return _get_ray(s, t, riud, rf);
  }

  Vec3 origin;
  Vec3 lower_left_corner;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 u, v, w;
  float lens_radius;
  float time0, time1;
};
