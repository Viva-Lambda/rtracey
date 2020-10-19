#pragma once
#include <primitive.cuh>
#include <ray.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

__host__ __device__ void mkSolidColor(Primitive &p,
                                      Color c) {
  p.ttype = SOLID_COLOR;
  p.tp1x = c.x();
  p.tp1y = c.y();
  p.tp1z = c.z();
}

__host__ __device__ void mkCheckTex(Primitive &p, Color c) {
  p.ttype = CHECKER;
  p.tp1x = c.x();
  p.tp1y = c.y();
  p.tp1z = c.z();
}

__host__ __device__ void
mkImageTex(Primitive &p, int w, int h, int bpp, int idx) {
  p.ttype = IMAGE;
  p.width = w;
  p.height = h;
  p.bytes_per_pixel = bpp;
  p.index = idx;
}
__device__ void mkNoiseTex(Primitive &p, float s,
                           curandState *loc) {
  p.ttype = NOISE;
  p.scale = s;
  p.loc = loc;
}

__host__ __device__ void mkLambert(Primitive &p) {
  p.mtype = LAMBERTIAN;
}
__host__ __device__ void mkMetal(Primitive &p, float f) {
  p.mtype = METAL;
  p.fuzz_ref_idx = f;
}
__host__ __device__ void mkDielectric(Primitive &p,
                                      float f) {
  p.mtype = DIELECTRIC;
  p.fuzz_ref_idx = f;
}
__host__ __device__ void mkDiffuseLight(Primitive &p) {
  p.mtype = DIFFUSE_LIGHT;
}
__host__ __device__ void mkIsotropic(Primitive &p) {
  p.mtype = ISOTROPIC;
}
__host__ __device__ void mkSphere(Primitive &p, Point3 c,
                                  float radius) {
  p.p1x = c.x();
  p.p1y = c.y();
  p.p1z = c.z();
  p.htype = SPHERE;
  p.radius = radius;
}
__host__ __device__ void mkMovingSphere(Primitive &p,
                                        Point3 c1,
                                        Point3 c2,
                                        float radius) {
  p.p1x = c1.x();
  p.p1y = c1.y();
  p.p1z = c1.z();
  p.p2x = c2.x();
  p.p2y = c2.y();
  p.p2z = c2.z();
  p.htype = MOVING_SPHERE;
  p.radius = radius;
}
__host__ __device__ void mkRect(Primitive &p, float a0,
                                float a1, float b0,
                                float b1, float k, float nx,
                                float ny, float nz) {
  p.htype = RECTANGLE;
  p.p1x = a0;
  p.p1y = a1;
  p.p2x = b0;
  p.p2y = b1;
  p.radius = k;
  p.n1x = nx;
  p.n1y = ny;
  p.n1z = nz;
}
__host__ __device__ void mkXyRect(Primitive &p, float x0,
                                  float x1, float y0,
                                  float y1, float k) {
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 1.0f;
  mkRect(p, x0, x1, y0, y1, k, nx, ny, nz);
  p.htype = XY_RECT;
}
__host__ __device__ void mkXzRect(Primitive &p, float x0,
                                  float x1, float z0,
                                  float z1, float k) {
  float nx = 0.0f;
  float ny = 1.0f;
  float nz = 0.0f;
  mkRect(p, x0, x1, z0, z1, k, nx, ny, nz);
  p.htype = XZ_RECT;
}
__host__ __device__ void mkYzRect(Primitive &p, float y0,
                                  float y1, float z0,
                                  float z1, float k) {
  float nx = 1.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  mkRect(p, y0, y1, z0, z1, k, nx, ny, nz);
  p.htype = YZ_RECT;
}

__host__ __device__ void mkTriangle(Primitive &p, float x0,
                                    float y0, float z0,
                                    float x1, float y1,
                                    float z1, float x2,
                                    float y2, float z2) {
  p.htype = TRIANGLE;
  p.p1x = x0;
  p.p1y = y0;
  p.p1z = z0;

  p.p2x = x1;
  p.p2y = y1;
  p.p2z = z1;

  p.n1x = x2;
  p.n1y = y2;
  p.n1z = z2;
}
__host__ __device__ void
mkAaTriangle(Primitive &p, float a0, float a1, float a2,
             float b0, float b1, float k, float nx,
             float ny, float nz) {
  p.htype = TRIANGLE;
  p.n1x = nx;
  p.n1y = ny;
  p.n1z = nz;
  p.p1x = a0;
  p.p1y = a1;
  p.p1z = a2;
  p.p2x = b0;
  p.p2y = b1;
  p.p2z = k;
}
__host__ __device__ void mkXyTriangle(Primitive &p,
                                      float x0, float x1,
                                      float x2, float y0,
                                      float y1, float z) {
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 1.0f;
  mkAaTriangle(p, x0, x1, x2, y0, y1, z, nx, ny, nz);
  p.htype = XY_TRIANGLE;
}
__host__ __device__ void mkXzTriangle(Primitive &p,
                                      float x0, float x1,
                                      float x2, float z0,
                                      float z1, float y) {
  float nx = 0.0f;
  float ny = 1.0f;
  float nz = 0.0f;
  mkAaTriangle(p, x0, x1, x2, z0, z1, y, nx, ny, nz);
  p.htype = XZ_TRIANGLE;
}
__host__ __device__ void mkYzTriangle(Primitive &p,
                                      float y0, float y1,
                                      float y2, float z0,
                                      float z1, float x) {
  float nx = 1.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  mkAaTriangle(p, y0, y1, y2, z0, z1, x, nx, ny, nz);
  p.htype = YZ_TRIANGLE;
}
