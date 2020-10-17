#pragma once
// shade utils
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <utils.cuh>
#include <vec3.cuh>

__host__ __device__ void get_sphere_uv(const Vec3 &p,
                                       float &u, float &v) {
  auto phi = atan2(p.z(), p.x());
  auto theta = asin(p.y());
  u = 1 - (phi + M_PI) / (2 * M_PI);
  v = (theta + M_PI / 2) / M_PI;
}

template <TextureType t>
__host__ __device__ Color
color_value(const SceneObjects &s, const HitRecord &rec) {
  return Color(0.0f);
}
template <>
__host__ __device__ Color color_value<SOLID_TET>(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  int tp1x = s.tp1xs[prim_idx];
  int tp1y = s.tp1ys[prim_idx];
  int tp1z = s.tp1zs[prim_idx];
  return Color(tp1x, tp1y, tp1z);
}
template <>
__host__ __device__ Color color_value<CHECKER_TET>(
    const SceneObjects &s, const HitRecord &rec) {
  Point3 p = rec.p;
  Color odd = color_value<SOLID_TET>(s, rec);
  Color even = 1 - odd;
  float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                sin(10.0f * p.z());
  if (sines < 0) {
    return odd;
  } else {
    return even;
  }
}
template <>
__host__ __device__ Color color_value<NOISE_TET>(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  Point3 p = rec.p;
  float scale = s.scales[prim_idx];
  float zscale = scale * p.z();
  Perlin noise(s.rand);
  float turbulance = 10.0f * noise.turb(p);
  Color white(1.0f, 1.0f, 1.0f);
  return white * 0.5f * (1.0f + sin(zscale + turbulance));
}
template <>
__host__ __device__ Color color_value<IMAGE_TET>(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  int u = rec.u;
  int v = rec.v;
  int width = s.widths[prim_idx];
  int height = s.heights[prim_idx];
  int bpp = s.bytes_per_pixels[prim_idx];
  int idx = s.image_indices[prim_idx];

  if (s.tdata == nullptr) {
    return Color(1.0, 0.0, 0.0);
  }
  u = clamp(u, 0.0, 1.0);
  v = 1.0 - clamp(v, 0.0, 1.0); // flip v to im coords
  int w = width;
  int h = height;
  int xi = (int)(u * w);
  int yj = (int)(v * h);
  xi = xi >= w ? w - 1 : xi;
  yj = yj >= h ? h - 1 : yj;
  int bytes_per_line = bpp * w;
  int pixel = yj * bytes_per_line + xi * bpp + idx;
  Color c(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < bpp; i++) {
    float pixel_val =
        static_cast<float>(s.tdata[pixel + i]);
    c[i] = pixel_val / 255;
  }
  return c;
}
template <>
__host__ __device__ Color color_value<TEXTURE>(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  TextureType ttype =
      static_cast<TextureType>(s.ttypes[prim_idx]);
  Color c(0.0f);
  if (ttype == NONE_TET) {
    return c;
  } else if (ttype == SOLID_TET) {
    c = color_value<SOLID_TET>(s, rec);
  } else if (ttype == CHECKER_TET) {
    c = color_value<CHECKER_TET>(s, rec);
  } else if (ttype == NOISE_TET) {
    c = color_value<NOISE_TET>(s, rec);
  } else if (ttype == IMAGE_TET) {
    c = color_value<IMAGE_TET>(s, rec);
  }
  return c;
}
template <MaterialType m>
__host__ __device__ bool
scatter(const SceneObjects &s, const Ray &r,
        const HitRecord &rec, Color &attenuation,
        Ray &r_out, float &pdf, curandState *loc) {
  return false;
}
template <>
__device__ bool
scatter<LAMBERTIAN_MAT>(const SceneObjects &s, const Ray &r,
                        const HitRecord &rec,
                        Color &attenuation, Ray &r_out,
                        float &pdf, curandState *loc) {
  Onb uvw;
  uvw.build_from_w(rec.normal);
  auto direction = uvw.local(random_cosine_direction(loc));
  r_out = Ray(rec.p, to_unit(direction), r.time());
  attenuation = color_value<TEXTURE>(s, rec);
  pdf = dot(uvw.w(), r_out.direction()) / M_PI;
  return true;
}
template <>
__device__ bool
scatter<METAL_MAT>(const SceneObjects &s, const Ray &r,
                   const HitRecord &rec, Color &attenuation,
                   Ray &r_out, float &pdf,
                   curandState *loc) {
  int prim_idx = rec.primitive_index;
  float fuzz = s.fuzz_ref_idxs[prim_idx];
  Vec3 reflected =
      reflect(to_unit(r.direction()), rec.normal);
  r_out = Ray(rec.p,
              reflected + fuzz * random_in_unit_sphere(loc),
              r.time());
  attenuation = color_value<TEXTURE>(s, rec);
  pdf = 1.0f;
  return (dot(r_out.direction(), rec.normal) > 0.0f);
}
template <>
__device__ bool
scatter<DIELECTRIC_MAT>(const SceneObjects &s, const Ray &r,
                        const HitRecord &rec,
                        Color &attenuation, Ray &r_out,
                        float &pdf, curandState *loc) {
  int prim_idx = rec.primitive_index;
  float ref_idx = s.fuzz_ref_idxs[prim_idx];
  pdf = 1.0f;
  Vec3 outward_normal;
  Vec3 reflected = reflect(r.direction(), rec.normal);
  float ni_over_nt;
  attenuation = Vec3(1.0f, 1.0f, 1.0f);
  Vec3 refracted;
  float reflect_prob;
  float cosine;
  if (dot(r.direction(), rec.normal) > 0.0f) {
    outward_normal = -rec.normal;
    ni_over_nt = ref_idx;
    cosine = dot(r.direction(), rec.normal) /
             r.direction().length();
    cosine = sqrt(1.0f -
                  ni_over_nt * ni_over_nt *
                      (1 - cosine * cosine));
  } else {
    outward_normal = rec.normal;
    ni_over_nt = 1.0f / ref_idx;
    cosine = -dot(r.direction(), rec.normal) /
             r.direction().length();
  }
  if (refract(r.direction(), outward_normal, ni_over_nt,
              refracted))
    reflect_prob = fresnelCT(cosine, ref_idx);
  else
    reflect_prob = 1.0f;
  if (curand_uniform(loc) < reflect_prob)
    r_out = Ray(rec.p, reflected, r.time());
  else
    r_out = Ray(rec.p, refracted, r.time());
  return true;
}
template <>
__device__ bool scatter<DIFFUSE_LIGHT_MAT>(
    const SceneObjects &s, const Ray &r,
    const HitRecord &rec, Color &attenuation, Ray &r_out,
    float &pdf, curandState *loc) {
  pdf = 1.0f;
  return false;
}
template <>
__device__ bool
scatter<ISOTROPIC_MAT>(const SceneObjects &s, const Ray &r,
                       const HitRecord &rec,
                       Color &attenuation, Ray &r_out,
                       float &pdf, curandState *loc) {
  r_out = Ray(rec.p, random_in_unit_sphere(loc), r.time());
  attenuation = color_value<TEXTURE>(s, rec);
  pdf = 1.0f;
  return true;
}
template <>
__device__ bool
scatter<MATERIAL>(const SceneObjects &s, const Ray &r,
                  const HitRecord &rec, Color &attenuation,
                  Ray &r_out, float &pdf,
                  curandState *loc) {
  int prim_idx = rec.primitive_index;
  bool res = false;
  MaterialType mtype =
      static_cast<MaterialType>(s.mtypes[prim_idx]);
  if (mtype == LAMBERTIAN_MAT) {
    res = scatter<LAMBERTIAN_MAT>(s, r, rec, attenuation,
                                  r_out, pdf, loc);
  } else if (mtype == METAL_MAT) {
    res = scatter<METAL_MAT>(s, r, rec, attenuation, r_out,
                             pdf, loc);
  } else if (mtype == DIELECTRIC_MAT) {
    res = scatter<METAL_MAT>(s, r, rec, attenuation, r_out,
                             pdf, loc);
  } else if (mtype == DIFFUSE_LIGHT_MAT) {
    res = scatter<DIFFUSE_LIGHT_MAT>(s, r, rec, attenuation,
                                     r_out, pdf, loc);
  } else if (mtype == ISOTROPIC_MAT) {
    res = scatter<ISOTROPIC_MAT>(s, r, rec, attenuation,
                                 r_out, pdf, loc);
  }
  return res;
}

template <MaterialType m>
__device__ Color emitted(const SceneObjects &s,
                         const HitRecord &rec) {
  return Color(0.0f);
}

template <>
__device__ Color emitted<DIFFUSE_LIGHT_MAT>(
    const SceneObjects &s, const HitRecord &rec) {
  return color_value<TEXTURE>(s, rec);
}

template <MaterialType m>
__device__ float
scattering_pdf(const SceneObjects &s, const Ray &r_in,
               const HitRecord &rec, const Ray &r_out) {
  return 0.0f;
}

template <>
__device__ float scattering_pdf<LAMBERTIAN_MAT>(
    const SceneObjects &s, const Ray &r_in,
    const HitRecord &rec, const Ray &r_out) {
  auto cosine = dot(rec.normal, to_unit(r_out.direction()));
  return cosine < 0 ? 0 : cosine / M_PI;
}
