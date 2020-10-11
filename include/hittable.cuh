#pragma once

#include <aabb.cuh>
#include <external.hpp>
#
#include <ray.cuh>
#include <vec3.cuh>

class Material;

/**
 @brief Records information about the ray hit.

 The information recorded in here is later used in
 scattering code. Basically what we have here should
 be sufficient for evaluating a bidirectional surface
 distribution reflection function.

 @param t distance to hit point location
 @param p the point that is hit by the ray
 @param normal the normal to the hit point.
 @param mat_ptr material attributed to hit point.
 @param u normalized screen space u coordinate
 @param v normalized screen space v coordinate
 @param front_face check which indicates the direction
 of the normal
 */
struct HitRecord {
  float t;
  Point3 p;
  Vec3 normal;
  Material *mat_ptr;
  float u, v;
  bool front_face;

  /**
    @brief check if ray hits the front side of the object

    We check the angle between the incoming ray and
    hit point. When vectors point in same direction (not
    necessarily parallel)
    their dot product is positive.

    @param r incoming ray
    @param norm the surface normal of the hit point.
   */
  __host__ __device__ void
  set_front_face(const Ray &r, const Vec3 &norm) {
    front_face = dot(r.direction(), norm) < 0.0f;
    normal = front_face ? norm : -norm;
  }
};

/**
  @brief A Scene Object that can be hit by ray.

  Hittable is the base class for all objects that are going
  to be traversed and hit by our algorithm. As opposed to
  those that are for sure not going to be hit, like sky,
  sun, etc.

  @param center is not necessary but represents
  the center of the bounding box of the object.

  @param mat_ptr material of the object that contributes to
  the bsdf.
 */
class Hittable {
public:
  Material *mat_ptr;
  Point3 center;

public:
  /**
    @brief determine if this object is hit by the incoming
    ray

    @param r incoming ray.
    @param d_min minimum distance.
    @param d_max maximum distance.
    @param rec the record object that would hold the
    parameters required by the scattering function later on.
   */
  __device__ virtual bool hit(const Ray &r, float d_min,
                              float d_max,
                              HitRecord &rec) const = 0;

  /**
    @brief determine the bounding box of this object.

    Compute the bounding box of a given object.
    For moving objects the time information is necessary
    since their bounding box covers the whole space between
    their traveled time period.

    @param t0 time0
    @param t1 time1
   */
  __host__ __device__ virtual bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const = 0;
  __device__ virtual float
  pdf_value(const Point3 &o, const Point3 &v) const {
    return 0.0f;
  }

  __device__ virtual Vec3 random(const Vec3 &v,
                                 curandState *loc) const {
    //
    return Vec3(1.0f, 0.0f, 0.0f);
  }
};
__device__ bool hit_to_hittables(Hittable **hs, int si,
                                 int ei, const Ray &r,
                                 float d_min, float d_max,
                                 HitRecord &rec) {
  //
  HitRecord temp;
  bool hit_anything = false;
  float closest_far = d_max;
  for (int i = si; i < ei; i++) {
    const Hittable *h = hs[i];
    bool isHit = h->hit(r, d_min, closest_far, temp);
    if (isHit == true) {
      hit_anything = true;
      closest_far = temp.t;
      rec = temp;
    }
  }
  return hit_anything;
}

__host__ __device__ bool
bounding_box_to_hittables(Hittable **hs, int si, int ei,
                          float t0, float t1,
                          Aabb &output_box) {
  if (ei == 0) {
    return false;
  }
  Aabb temp;
  bool first_box = true;
  for (int i = si; i < ei; i++) {
    const Hittable *h = hs[i];
    bool isBounding = h->bounding_box(t0, t1, temp);
    if (isBounding == false) {
      return false;
    }
    output_box = first_box
                     ? temp
                     : surrounding_box(output_box, temp);
    first_box = false;
    // center = output_box.center;
  }
  return true;
}

class Translate : public Hittable {
public:
  __host__ __device__ Translate(Hittable *p,
                                const Vec3 &displacement)
      : ptr(p), offset(displacement) {}

  __device__ bool hit(const Ray &r, float t_min,
                      float t_max,
                      HitRecord &rec) const override {
    Ray moved_r(r.origin() - offset, r.direction(),
                r.time());
    if (!ptr->hit(moved_r, t_min, t_max, rec))
      return false;

    rec.p += offset;
    rec.set_front_face(moved_r, rec.normal);

    return true;
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    if (!ptr->bounding_box(t0, t1, output_box))
      return false;

    output_box = Aabb(output_box.min() + offset,
                      output_box.max() + offset);

    return true;
  }

public:
  Hittable *ptr;
  Vec3 offset;
};
class RotateY : public Hittable {
public:
  __host__ __device__ RotateY(Hittable *p, float angle)
      : ptr(p) {
    float radians = degree_to_radian(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->bounding_box(0, 1, bbox);

    Point3 min(FLT_MAX);
    Point3 max(-FLT_MAX);

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          float x =
              i * bbox.max().x() + (1 - i) * bbox.min().x();
          float y =
              j * bbox.max().y() + (1 - j) * bbox.min().y();
          float z =
              k * bbox.max().z() + (1 - k) * bbox.min().z();

          float newx = cos_theta * x + sin_theta * z;
          float newz = -sin_theta * x + cos_theta * z;

          Vec3 tester(newx, y, newz);

          min = min_vec(min, tester);
          max = max_vec(max, tester);
        }
      }
    }
    bbox = Aabb(min, max);
  }

  __device__ bool hit(const Ray &r, float t_min,
                      float t_max,
                      HitRecord &rec) const override {
    Point3 origin = r.origin();
    Vec3 direction = r.direction();

    float x = (cos_theta * r.origin().x()) -
              (sin_theta * r.origin().z());

    float z = (sin_theta * r.origin().x()) +
              (cos_theta * r.origin().z());
    origin[0] = x;
    origin[2] = z;

    float dx = cos_theta * r.direction().x() -
               sin_theta * r.direction().z();

    float dz = sin_theta * r.direction().x() +
               cos_theta * r.direction().z();

    direction[0] = dx;
    direction[2] = dz;

    Ray rotated_r(origin, direction, r.time());

    if (!ptr->hit(rotated_r, t_min, t_max, rec))
      return false;

    Point3 p = rec.p;
    Vec3 normal = rec.normal;

    p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
    p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

    normal[0] = cos_theta * rec.normal[0] +
                sin_theta * rec.normal[2];
    normal[2] = -sin_theta * rec.normal[0] +
                cos_theta * rec.normal[2];

    rec.p = p;
    rec.set_front_face(rotated_r, normal);

    return true;
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    output_box = bbox;
    return hasbox;
  }

public:
  Hittable *ptr;
  float sin_theta;
  float cos_theta;
  bool hasbox;
  Aabb bbox;
};
class FlipFace : public Hittable {
public:
  __host__ __device__ FlipFace(Hittable *p) : ptr(p) {}

  __device__ bool hit(const Ray &r, float t_min,
                      float t_max,
                      HitRecord &rec) const override {

    if (!ptr->hit(r, t_min, t_max, rec))
      return false;

    rec.front_face = !rec.front_face;
    return true;
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    return ptr->bounding_box(t0, t1, output_box);
  }

public:
  Hittable *ptr;
};

__host__ __device__ int
farthest_index(Hittable **&hs, Hittable *&h, int nb_h) {
  //
  float max_dist = FLT_MIN;
  int max_dist_index = 0;
  Aabb tb;
  h->bounding_box(0.0f, 0.0f, tb);
  Point3 g_center = tb.center();
  for (int i = 0; i < nb_h; i++) {
    Aabb t;
    hs[i]->bounding_box(0.0f, 0.0f, t);
    Point3 scene_center = t.center();
    float dist = distance(g_center, scene_center);
    if (dist > max_dist) {
      max_dist = dist;
      max_dist_index = i;
    }
  }
  return max_dist_index;
}

__host__ __device__ void
swap_hit(Hittable *&h1, Hittable *&h2, Hittable *&temp) {
  temp = h1;
  h1 = h2;
  h2 = temp;
}

// implementing list structure from
// Foley et al. 2013, p. 1081
__host__ __device__ void order_scene(Hittable **&hs,
                                     int nb_h) {
  for (int i = 0; i < nb_h - 1; i += 2) {
    Hittable *h = hs[i];
    int fgi = farthest_index(hs, h, nb_h);
    Hittable *h1;
    swap_hit(hs[i + 1], hs[fgi], h1);
  }
}
