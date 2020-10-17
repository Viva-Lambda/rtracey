#pragma once
// container objects like box constant medium
#include <primitive.cuh>
#include <vec3.cuh>

struct Box {
  __host__ __device__ Box()
      : prims(nullptr), group_size(0) {}
  __host__ __device__ Box(const Primitive *sp,
                          const int gsize)
      : prims(sp), group_size(gsize) {}
  __host__ __device__ void minmax_points(Point3 &p1,
                                         Point3 &p2) const {
    Primitive p_1 = prims[0];
    Primitive p_2 = prims[1];

    float p0x = p_1.hparam.p1x; // p0x
    float p1x = p_1.hparam.p1y; // p1x
    float p0y = p_1.hparam.p2x; // p0y
    float p1y = p_1.hparam.p2y; // p1y

    float p0z = p_2.hparam.radius; // p0z
    float p1z = p_1.hparam.radius; // p1z

    p1 = Point3(p0x, p0y, p0z);
    p2 = Point3(p1x, p1y, p1z);
  }

  const Primitive *prims;
  const int group_size;
};
struct ConstantMedium {
  __host__ __device__ ConstantMedium(const Primitive *b,
                                     const float d,
                                     const TextureParam &a,
                                     const int gsize)
      : boundary(b), neg_inv_density(-1 / d),
        phase_function(mkIsotropicParam(a)),
        nb_prims(gsize) {}

  __device__ ConstantMedium(const Primitive *b,
                            const float d,
                            const TextureParam &a,
                            const int gsize, curandState *s)
      : boundary(b), neg_inv_density(-1 / d),
        phase_function(mkIsotropicParam(a)), rState(s),
        nb_prims(gsize) {}

  const Primitive *boundary;
  MaterialParam phase_function;
  const float neg_inv_density;
  curandState *rState;
  const int nb_prims;
};
struct SimpleMesh {
  __host__ __device__ SimpleMesh()
      : prims(nullptr), group_size(0) {}
  __host__ __device__ SimpleMesh(const Primitive *prm,
                                 const int gsize)
      : prims(prm), group_size(gsize) {}
  const Primitive *prims;
  const int group_size;
};
