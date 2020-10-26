#pragma once
// acceleration structure
#include <aabb.cuh>
#include <bbox.cuh>
#include <groupparam.cuh>
#include <hit.cuh>

__host__ __device__ int
farthest_index(GroupParam *gps, GroupParam g, int gs_idx) {
  float max_dist = FLT_MIN;
  int max_dist_index = 0;
  Aabb tb =
      Aabb(min_vec<GroupParam>(g), max_vec<GroupParam>(g));
  Point3 g_center = tb.center();
  for (int i = 0; i < gs_idx; i++) {
    Aabb t = Aabb(min_vec<GroupParam>(gps[i]),
                  max_vec<GroupParam>(gps[i]));
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
swap_hit(GroupParam &h1, GroupParam &h2, GroupParam &temp) {
  temp = h1;
  h1 = h2;
  h2 = temp;
}

// implementing list structure from
// Foley et al. 2013, p. 1081
__host__ __device__ void order_scene(GroupParam *gs,
                                     int nb_gs) {
  for (int i = 0; i < nb_gs - 1; i += 2) {
    GroupParam h = gs[i];
    int fgi = farthest_index(gs, h, nb_gs);
    GroupParam g1;
    swap_hit(gs[i + 1], gs[fgi], g1);
  }
}

// -------- BVH -------------------
/*

struct BvhNode {
  BvhNode *left;  // obje
  BvhNode *right; // obje
  Aabb box;

  __host__ BvhNode(){};
  __host__ BvhNode(GroupParam *objects, size_t start,
                   size_t end, float time1, float time2);
  __host__ BvhNode(std::vector<GroupParam> hlist,
                   float time1, float time2)
      : BvhNode(hlist.data(), 0, hlist.size(), time1,
                time2);
};

template <typename T>
bool hit(const T *node, const Ray &r_in, float tmin,
         float tmax, HitRecord &rec) {
  return false;
}
template <>
bool hit<BvhNode>(const BvhNode *bvh, const Ray &r_in,
                  float tmin, float tmax, HitRecord &rec) {
  //
  if (bvh->box.hit(r_in, tmin, tmax) == false) {
    return false;
  }
  bool hleft =
      hit<BvhNode>(bvh->left, r_in, tmin, tmax, record);
  float ntmax = hleft ? record.t : tmax;
  bool hright =
      hit<BvhNode>(bvh->right, r_in, tmin, ntmax, record);
  return hleft || hright;
}

bool bounding_box(double t1, double t2,
                  Aabb &output_bbox) const {
  output_bbox = box;
  return true;
};

bool box_compare(const GroupParam &a, const GroupParam &b,
                 int ax) {
  // kutulari eksenlerini kullanarak kontrol et
  Aabb ba;
  Aabb bb;
  if (!bounding_box<GroupParam>(a, ba) ||
      !bounding_box<GroupParam>(b, bb)) {
    std::cerr << "No bounding box in BVH node" << std::endl;
  }
  return ba.min()[ax] < bb.min()[ax];
}
bool box_compare_x(const GroupParam &a,
                   const GroupParam &b) {
  return box_compare(a, b, 0);
}
bool box_compare_y(const GroupParam &a,
                   const GroupParam &b) {
  return box_compare(a, b, 1);
}
bool box_compare_z(const GroupParam &a,
                   const GroupParam &b) {
  return box_compare(a, b, 2);
}
BvhNode::BvhNode(GroupParam *objects, size_t start,
                 size_t end, float time1, float time2) {
  int ax = h_random_int(0, 2);
  typedef bool comparator(const GroupParam,
                          const GroupParam);
  if (ax == 0) {
    comparator = box_compare_x;
  } else if (ax == 1) {
    comparator = box_compare_y;
  } else if (ax == 2) {
    comparator = box_compare_z;
  }
  size_t object_span = end - start;
  if (object_span == 1) {
    left = right = objects[start];
  } else if (object_span == 2) {
    if (comparator(objects[start], objects[start + 1])) {
      left = objects[start];
      right = objects[start + 1];
    } else {
      left = objects[start + 1];
      right = objects[start];
    }
  } else {
    std::sort(objects + start, objects + end, comparator);
    double mid = start + object_span / 2;
    left = new BvhNode(objects, start, mid, time1, time2);
    right = new BvhNode(objects, mid, end, time1, time2);
  }
  Aabb bleft;
  Aabb bright;
  if ((left->bounding_box(time1, time2, bleft) == false) ||
      (right->bounding_box(time1, time2, bright) ==
       false)) {
    std::cerr << "" << std::endl;
  }
  box = surrounding_box(bleft, bright);
}
*/
