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
template <TextureType t>
__host__ Color h_color_value(const SceneObjects &s,
                             const HitRecord &rec) {
  return Color(0.0f);
}
__host__ __device__ Color
solid_value(const SceneObjects &s, const HitRecord &rec) {
  int prim_idx;
  float tp1x, tp1y, tp1z;
  if (rec.group_scattering) {
    prim_idx = rec.group_index;
    tp1x = s.g_tp1xs[prim_idx];
    tp1y = s.g_tp1ys[prim_idx];
    tp1z = s.g_tp1zs[prim_idx];
  } else {
    prim_idx = rec.primitive_index;
    tp1x = s.tp1xs[prim_idx];
    tp1y = s.tp1ys[prim_idx];
    tp1z = s.tp1zs[prim_idx];
  }
  return Color(tp1x, tp1y, tp1z);
}
template <>
__device__ Color color_value<SOLID_COLOR>(
    const SceneObjects &s, const HitRecord &rec) {
  return solid_value(s, rec);
}
template <>
__host__ Color h_color_value<SOLID_COLOR>(
    const SceneObjects &s, const HitRecord &rec) {
  return solid_value(s, rec);
}
__host__ __device__ Color
checker_value(const SceneObjects &s, const HitRecord &rec,
              Color odd_val) {
  Point3 p = rec.p;
  Color odd = odd_val;
  Color even = 1.0f - odd_val;
  float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                sin(10.0f * p.z());
  if (sines < 0) {
    return odd;
  } else {
    return even;
  }
}
template <>
__device__ Color color_value<CHECKER>(
    const SceneObjects &s, const HitRecord &rec) {
  Color c = color_value<SOLID_COLOR>(s, rec);
  return checker_value(s, rec, c);
}
template <>
__host__ Color h_color_value<CHECKER>(
    const SceneObjects &s, const HitRecord &rec) {
  Color c = h_color_value<SOLID_COLOR>(s, rec);
  return checker_value(s, rec, c);
}
__host__ __device__ Color noise_value(const SceneObjects &s,
                                      const HitRecord &rec,
                                      float turb) {
  //
  int prim_idx;
  float scale;
  if (rec.group_scattering) {
    prim_idx = rec.primitive_index;
    scale = s.scales[prim_idx];
  } else {
    prim_idx = rec.group_index;
    scale = s.g_scales[prim_idx];
  }
  Point3 p = rec.p;
  float zscale = scale * p.z();
  float turbulance = 10.0f * turb;
  Color white(1.0f, 1.0f, 1.0f);
  return white * 0.5f * (1.0f + sin(zscale + turbulance));
}
template <>
__device__ Color color_value<NOISE>(const SceneObjects &s,
                                    const HitRecord &rec) {
  Perlin noise(s.rand);
  Point3 p = rec.p;
  float turb = noise.turb(p);
  return noise_value(s, rec, turb);
}
template <>
__host__ Color h_color_value<NOISE>(const SceneObjects &s,
                                    const HitRecord &rec) {
  Perlin noise(true);
  Point3 p = rec.p;
  float turb = noise.turb(p);
  return noise_value(s, rec, turb);
}
__host__ __device__ Color imcolor(const SceneObjects &s,
                                  const HitRecord &rec) {
  int width, height, bpp, idx, prim_idx;
  if (rec.group_scattering) {
    prim_idx = rec.primitive_index;
    width = s.widths[prim_idx];
    height = s.heights[prim_idx];
    bpp = s.bytes_per_pixels[prim_idx];
    idx = s.image_indices[prim_idx];
  } else {
    prim_idx = rec.group_index;
    width = s.g_widths[prim_idx];
    height = s.g_heights[prim_idx];
    bpp = s.g_bpps[prim_idx];
    idx = s.g_indices[prim_idx];
  }

  int u = rec.u;
  int v = rec.v;

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
__device__ Color color_value<IMAGE>(const SceneObjects &s,
                                    const HitRecord &rec) {
  return imcolor(s, rec);
}
template <>
__host__ Color h_color_value<IMAGE>(const SceneObjects &s,
                                    const HitRecord &rec) {
  return imcolor(s, rec);
}
template <>
__device__ Color color_value<TEXTURE>(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx;
  TextureType ttype;
  if (rec.group_scattering) {
    prim_idx = rec.group_index;
    ttype = static_cast<TextureType>(s.ttypes[prim_idx]);
  } else {
    prim_idx = rec.primitive_index;
    ttype = static_cast<TextureType>(s.g_ttypes[prim_idx]);
  }
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
template <>
__host__ Color h_color_value<TEXTURE>(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  TextureType ttype =
      static_cast<TextureType>(s.ttypes[prim_idx]);
  Color c(0.0f);
  if (ttype == NONE_TEXTURE) {
    return c;
  } else if (ttype == SOLID_COLOR) {
    c = h_color_value<SOLID_COLOR>(s, rec);
  } else if (ttype == CHECKER) {
    c = h_color_value<CHECKER>(s, rec);
  } else if (ttype == NOISE) {
    c = h_color_value<NOISE>(s, rec);
  } else if (ttype == IMAGE) {
    c = h_color_value<IMAGE>(s, rec);
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
