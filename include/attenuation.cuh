#pragma once
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <utils.cuh>
#include <vec3.cuh>

template <TextureType t>
__device__ Color color_value(const SceneObjects &s,
                             const HitRecord &rec) {
  return Color(0.0f);
}
template <>
__device__ Color color_value<SOLID_COLOR>(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  float tp1x = s.tp1xs[prim_idx];
  float tp1y = s.tp1ys[prim_idx];
  float tp1z = s.tp1zs[prim_idx];
  return Color(tp1x, tp1y, tp1z);
}
template <>
__device__ Color color_value<CHECKER>(
    const SceneObjects &s, const HitRecord &rec) {
  Point3 p = rec.p;
  Color odd = color_value<SOLID_COLOR>(s, rec);
  Color even = 1.0f - odd;
  float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                sin(10.0f * p.z());
  if (sines < 0) {
    return odd;
  } else {
    return even;
  }
}
template <>
__device__ Color color_value<NOISE>(const SceneObjects &s,
                                    const HitRecord &rec) {
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
__device__ Color color_value<IMAGE>(const SceneObjects &s,
                                    const HitRecord &rec) {
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
__device__ Color color_value<TEXTURE>(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  TextureType ttype =
      static_cast<TextureType>(s.ttypes[prim_idx]);
  Color c(0.0f);
  if (ttype == NONE_TEXTURE) {
    return c;
  } else if (ttype == SOLID_COLOR) {
    c = color_value<SOLID_COLOR>(s, rec);
  } else if (ttype == CHECKER) {
    c = color_value<CHECKER>(s, rec);
  } else if (ttype == NOISE) {
    c = color_value<NOISE>(s, rec);
  } else if (ttype == IMAGE) {
    c = color_value<IMAGE>(s, rec);
  }
  return c;
}

__host__ __device__ Color color_material(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  TextureType ttype =
      static_cast<TextureType>(s.ttypes[prim_idx]);
  float tp1x = s.tp1xs[prim_idx];
  float tp1y = s.tp1ys[prim_idx];
  float tp1z = s.tp1zs[prim_idx];
  return Color(tp1x, tp1y, tp1z);
}
