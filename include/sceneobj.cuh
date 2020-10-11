#ifndef SCENEOBJ_CUH
#define SCENEOBJ_CUH
#include <debug.hpp>
#include <ray.cuh>
#include <scenegroup.cuh>
#include <sceneparam.cuh>
#include <sceneprim.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct SceneObjects {
  // texture params
  int *ttypes;
  Color *cvals;
  float *scales;

  // image params
  int *widths;
  int *heights;
  int *bytes_per_pixels;
  int *image_indices;
  // unsigned char *tdata;
  // int tsize;

  // material params
  int *mtypes;
  float *fuzz_ref_idxs;

  // hittable params
  int *htypes;
  float *p1xs, *p1ys, *p1zs;
  float *p2xs, *p2ys, *p2zs;
  float *rads;
  float *n1xs, *n1ys, *n1zs;

  // group params
  int *prim_group_indices;

  //
  int *group_starts;
  int *group_sizes;
  int *group_ids;
  int *gtypes;
  float *g_densities;
  // group texture params
  int *g_ttypes;
  float *g_tp1xs, *g_tp1ys, *g_tp1zs;
  float *g_scales;
  int *g_widths, *g_heights, *g_bpps, *g_indices;

  int nb_groups;
  int nb_prims;

  __host__ __device__ SceneObjects() {}
  __host__ __device__ SceneObjects(SceneGroup *gs, int nb_g)
      : nb_groups(nb_g) {
    alloc_group_params(nb_g);
    //
    int nb_prim = 0;
    for (int i = 0; i < nb_groups; i++) {
      SceneGroup g = gs[i];
      nb_prim += g.group_size;
    }
    nb_prims = nb_prim;
    alloc_prim_params(nb_prim);

    for (int i = 0; i < nb_groups; i++) {
      SceneGroup g = gs[i];
      group_sizes[i] = g.group_size;
      group_ids[i] = g.group_id;
      gtypes[i] = g.gtype;
      int gstart = i * g.group_size;
      group_starts[i] = gstart == 0 ? g.group_size : gstart;

      for (int j = 0; j < g.group_size; j++) {
        int gindex = gstart + j;
        ScenePrim prim;
        bool is_valid_index = g.get(j, prim);
        if (is_valid_index) {
          set_primitive(prim, gindex);
        }
      }
    }
  }
  __host__ __device__ void set_primitive(ScenePrim &p,
                                         int gindex) {
    ttypes[gindex] = p.mparam.tparam.ttype;
    cvals[gindex] = p.mparam.tparam.cval;
    scales[gindex] = p.mparam.tparam.scale;
    widths[gindex] = p.mparam.tparam.imp.width;
    heights[gindex] = p.mparam.tparam.imp.height;
    bytes_per_pixels[gindex] =
        p.mparam.tparam.imp.bytes_per_pixel;
    image_indices[gindex] = p.mparam.tparam.imp.index;
    mtypes[gindex] = p.mparam.mtype;
    fuzz_ref_idxs[gindex] = p.mparam.fuzz_ref_idx;
    htypes[gindex] = p.hparam.htype;

    p1xs[gindex] = p.hparam.p1x;
    p1ys[gindex] = p.hparam.p1y;
    p1zs[gindex] = p.hparam.p1z;

    p2xs[gindex] = p.hparam.p2x;
    p2ys[gindex] = p.hparam.p2y;
    p2zs[gindex] = p.hparam.p2z;

    n1xs[gindex] = p.hparam.n1x;
    n1ys[gindex] = p.hparam.n1y;
    n1zs[gindex] = p.hparam.n1z;
    rads[gindex] = p.hparam.radius;
    prim_group_indices[gindex] = p.group_index;
  }
  __host__ __device__ void alloc_group_params(int nb_g) {
    group_starts = new int[nb_g];
    group_sizes = new int[nb_g];
    group_ids = new int[nb_g];
    gtypes = new int[nb_g];
    g_densities = new float[nb_g];
    g_ttypes = new int[nb_g];
    g_tp1xs = new float[nb_g];
    g_tp1ys = new float[nb_g];
    g_tp1zs = new float[nb_g];
    g_scales = new float[nb_g];
    g_widths = new int[nb_g];
    g_heights = new int[nb_g];
    g_bpps = new int[nb_g];
    g_indices = new int[nb_g];
  }
  __host__ __device__ void alloc_prim_params(int nb_ps) {
    ttypes = new int[nb_ps];
    cvals = new Color[nb_ps];
    scales = new float[nb_ps];
    widths = new int[nb_ps];
    heights = new int[nb_ps];
    bytes_per_pixels = new int[nb_ps];
    image_indices = new int[nb_ps];
    mtypes = new int[nb_ps];
    fuzz_ref_idxs = new float[nb_ps];
    htypes = new int[nb_ps];

    p1xs = new float[nb_ps];
    p1ys = new float[nb_ps];
    p1zs = new float[nb_ps];

    p2xs = new float[nb_ps];
    p2ys = new float[nb_ps];
    p2zs = new float[nb_ps];

    rads = new float[nb_ps];

    n1xs = new float[nb_ps];
    n1ys = new float[nb_ps];
    n1zs = new float[nb_ps];

    prim_group_indices = new int[nb_ps];
  }
  __host__ __device__ TextureParam
  get_group_texture_param(int gindex) const {
    Color cv(g_tp1xs[gindex], g_tp1ys[gindex],
             g_tp1zs[gindex]);
    ImageParam img_p(g_widths[gindex], g_heights[gindex],
                     g_bpps[gindex], g_indices[gindex]);
    TextureParam tpr(
        static_cast<TextureType>(g_ttypes[gindex]), cv,
        g_scales[gindex], img_p);
    return tpr;
  }
  __host__ __device__ void fill_group(SceneGroup &group,
                                      int gindex) const {
    group.gtype = static_cast<GroupType>(gtypes[gindex]);
    group.group_size = group_sizes[gindex];
    group.group_id = group_ids[gindex];
    group.density = g_densities[gindex];
    group.tparam = get_group_texture_param(gindex);
    ScenePrim *prims = new ScenePrim[group.group_size];
    int gstart = group_starts[gindex];
    for (int i = 0; i < group.group_size; i++) {
      gstart += i;
      prims[i] = get_primitive(gstart, group.group_id);
    }
  }
  __host__ SceneObjects to_device() {
    SceneObjects sobjs;
    sobjs.nb_prims = nb_prims;
    sobjs.nb_groups = nb_groups;

    thrust::device_ptr<int> d_ttypes;
    upload_thrust<int>(d_ttypes, ttypes, nb_prims);
    sobjs.ttypes = thrust::raw_pointer_cast(d_ttypes);

    thrust::device_ptr<Color> d_cvals;
    upload_thrust<Color>(d_cvals, cvals, nb_prims);
    sobjs.cvals = thrust::raw_pointer_cast(d_cvals);

    thrust::device_ptr<float> d_scales;
    upload_thrust<float>(d_scales, scales, nb_prims);
    sobjs.scales = thrust::raw_pointer_cast(d_scales);

    thrust::device_ptr<int> d_widths;
    upload_thrust<int>(d_widths, widths, nb_prims);
    sobjs.widths = thrust::raw_pointer_cast(d_widths);

    thrust::device_ptr<int> d_heights;
    upload_thrust<int>(d_heights, heights, nb_prims);
    sobjs.heights = thrust::raw_pointer_cast(d_heights);

    thrust::device_ptr<int> d_bpps;
    upload_thrust<int>(d_bpps, bytes_per_pixels, nb_prims);
    sobjs.bytes_per_pixels =
        thrust::raw_pointer_cast(d_bpps);

    thrust::device_ptr<int> d_img_indices;
    upload_thrust<int>(d_img_indices, image_indices,
                       nb_prims);
    sobjs.image_indices =
        thrust::raw_pointer_cast(d_img_indices);

    thrust::device_ptr<int> d_mtypes;
    upload_thrust<int>(d_mtypes, mtypes, nb_prims);
    sobjs.mtypes = thrust::raw_pointer_cast(d_mtypes);

    thrust::device_ptr<float> d_fuzzs;
    upload_thrust<float>(d_fuzzs, fuzz_ref_idxs, nb_prims);
    sobjs.fuzz_ref_idxs = thrust::raw_pointer_cast(d_fuzzs);

    thrust::device_ptr<int> d_htypes;
    upload_thrust<int>(d_htypes, htypes, nb_prims);
    sobjs.htypes = thrust::raw_pointer_cast(d_htypes);

    thrust::device_ptr<float> d_p1xs;
    upload_thrust<float>(d_p1xs, p1xs, nb_prims);
    sobjs.p1xs = thrust::raw_pointer_cast(d_p1xs);

    thrust::device_ptr<float> d_p1ys;
    upload_thrust<float>(d_p1ys, p1ys, nb_prims);
    sobjs.p1ys = thrust::raw_pointer_cast(d_p1ys);

    thrust::device_ptr<float> d_p1zs;
    upload_thrust<float>(d_p1zs, p1zs, nb_prims);
    sobjs.p1zs = thrust::raw_pointer_cast(d_p1zs);

    thrust::device_ptr<float> d_p2xs;
    upload_thrust<float>(d_p2xs, p2xs, nb_prims);
    sobjs.p2xs = thrust::raw_pointer_cast(d_p2xs);

    thrust::device_ptr<float> d_p2ys;
    upload_thrust<float>(d_p2ys, p2ys, nb_prims);
    sobjs.p2ys = thrust::raw_pointer_cast(d_p2ys);

    thrust::device_ptr<float> d_p2zs;
    upload_thrust<float>(d_p2zs, p2zs, nb_prims);
    sobjs.p2zs = thrust::raw_pointer_cast(d_p2zs);

    thrust::device_ptr<float> d_n1xs;
    upload_thrust<float>(d_n1xs, n1xs, nb_prims);
    sobjs.n1xs = thrust::raw_pointer_cast(d_n1xs);

    thrust::device_ptr<float> d_n1ys;
    upload_thrust<float>(d_n1ys, n1ys, nb_prims);
    sobjs.n1ys = thrust::raw_pointer_cast(d_n1ys);

    thrust::device_ptr<float> d_n1zs;
    upload_thrust<float>(d_n1zs, n1zs, nb_prims);
    sobjs.n1zs = thrust::raw_pointer_cast(d_n1zs);

    thrust::device_ptr<float> d_rads;
    upload_thrust<float>(d_rads, rads, nb_prims);
    sobjs.rads = thrust::raw_pointer_cast(d_rads);

    thrust::device_ptr<int> d_prim_g_indices;
    upload_thrust<int>(d_prim_g_indices, prim_group_indices,
                       nb_prims);
    sobjs.prim_group_indices =
        thrust::raw_pointer_cast(d_prim_g_indices);

    thrust::device_ptr<int> d_gstarts;
    upload_thrust<int>(d_gstarts, group_starts, nb_groups);
    sobjs.group_starts =
        thrust::raw_pointer_cast(d_gstarts);

    thrust::device_ptr<int> d_gsizes;
    upload_thrust<int>(d_gsizes, group_sizes, nb_groups);
    sobjs.group_sizes = thrust::raw_pointer_cast(d_gsizes);

    thrust::device_ptr<int> d_gids;
    upload_thrust<int>(d_gids, group_ids, nb_groups);
    sobjs.group_ids = thrust::raw_pointer_cast(d_gids);

    thrust::device_ptr<int> d_gts;
    upload_thrust<int>(d_gts, gtypes, nb_groups);
    sobjs.gtypes = thrust::raw_pointer_cast(d_gts);

    thrust::device_ptr<float> d_g_dens;
    upload_thrust<float>(d_g_dens, g_densities, nb_groups);
    sobjs.g_densities = thrust::raw_pointer_cast(d_g_dens);

    thrust::device_ptr<int> d_g_ttypes;
    upload_thrust<int>(d_g_ttypes, g_ttypes, nb_groups);
    sobjs.g_ttypes = thrust::raw_pointer_cast(d_g_ttypes);

    thrust::device_ptr<float> d_g_tp1xs;
    upload_thrust<float>(d_g_tp1xs, g_tp1xs, nb_groups);
    sobjs.g_tp1xs = thrust::raw_pointer_cast(d_g_tp1xs);

    thrust::device_ptr<float> d_g_tp1ys;
    upload_thrust<float>(d_g_tp1ys, g_tp1ys, nb_groups);
    sobjs.g_tp1ys = thrust::raw_pointer_cast(d_g_tp1ys);

    thrust::device_ptr<float> d_g_tp1zs;
    upload_thrust<float>(d_g_tp1zs, g_tp1zs, nb_groups);
    sobjs.g_tp1zs = thrust::raw_pointer_cast(d_g_tp1zs);

    thrust::device_ptr<float> d_g_scales;
    upload_thrust<float>(d_g_scales, g_scales, nb_groups);
    sobjs.g_scales = thrust::raw_pointer_cast(d_g_scales);

    thrust::device_ptr<int> d_g_widths;
    upload_thrust<int>(d_g_widths, g_widths, nb_groups);
    sobjs.g_widths = thrust::raw_pointer_cast(d_g_widths);

    thrust::device_ptr<int> d_g_heights;
    upload_thrust<int>(d_g_heights, g_heights, nb_groups);
    sobjs.g_heights = thrust::raw_pointer_cast(d_g_heights);

    thrust::device_ptr<int> d_g_bpps;
    upload_thrust<int>(d_g_bpps, g_bpps, nb_groups);
    sobjs.g_bpps = thrust::raw_pointer_cast(d_g_bpps);

    thrust::device_ptr<int> d_g_indices;
    upload_thrust<int>(d_g_indices, g_indices, nb_groups);
    sobjs.g_indices = thrust::raw_pointer_cast(d_g_indices);

    return sobjs;
  }
  __host__ void d_free() {
    cudaFree(ttypes);
    cudaFree(cvals);
    cudaFree(scales);
    cudaFree(widths);
    cudaFree(heights);
    cudaFree(bytes_per_pixels);
    cudaFree(image_indices);
    // cudaFree(tdata);
    cudaFree(mtypes);
    cudaFree(fuzz_ref_idxs);
    cudaFree(htypes);

    cudaFree(p1xs);
    cudaFree(p1ys);
    cudaFree(p1zs);

    cudaFree(p2xs);
    cudaFree(p2ys);
    cudaFree(p2zs);

    cudaFree(n1xs);
    cudaFree(n1ys);
    cudaFree(n1zs);

    cudaFree(rads);
    cudaFree(prim_group_indices);

    cudaFree(group_starts);
    cudaFree(group_sizes);
    cudaFree(group_ids);
    cudaFree(gtypes);
    cudaFree(g_densities);
    cudaFree(g_ttypes);
    cudaFree(g_tp1xs);
    cudaFree(g_tp1ys);
    cudaFree(g_tp1zs);
    cudaFree(g_scales);
    cudaFree(g_widths);
    cudaFree(g_heights);
    cudaFree(g_bpps);
    cudaFree(g_indices);
  }
  __host__ __device__ ScenePrim
  get_primitive(int gstart, int group_id) const {
    HittableParam ht(
        static_cast<HittableType>(htypes[gstart]),
        p1xs[gstart], p1ys[gstart], p1zs[gstart],
        p2xs[gstart], p2ys[gstart], p2zs[gstart],
        n1xs[gstart], n1ys[gstart], n1zs[gstart],
        rads[gstart]);
    ImageParam imp(widths[gstart], heights[gstart],
                   bytes_per_pixels[gstart],
                   image_indices[gstart]);
    TextureParam tp(
        static_cast<TextureType>(ttypes[gstart]),
        cvals[gstart], scales[gstart], imp);
    MaterialParam mp(
        tp, static_cast<MaterialType>(mtypes[gstart]),
        fuzz_ref_idxs[gstart]);
    ScenePrim prim(mp, ht, prim_group_indices[gstart],
                   group_id);
    return prim;
  }
  __host__ __device__ void
  mk_hittable_list(Hittable **&hs) {
    for (int i = 0; i < nb_groups; i++) {
      SceneGroup group;
      fill_group(group, i);
      HittableGroup hg = group.to_hittable();
      hs[i] = static_cast<Hittable *>(&hg);
    }
  }
  __host__ __device__ void
  mk_hittable_list(Hittable **&hs, unsigned char *&td) {
    for (int i = 0; i < nb_groups; i++) {
      SceneGroup group;
      fill_group(group, i);
      HittableGroup hg = group.to_hittable(td);
      hs[i] = static_cast<Hittable *>(&hg);
    }
  }
  __device__ void mk_hittable_list(Hittable **&hs,
                                   unsigned char *&td,
                                   curandState *loc) {
    for (int i = 0; i < nb_groups; i++) {
      SceneGroup group;
      fill_group(group, i);
      HittableGroup hg = group.to_hittable(td, loc);
      hs[i] = static_cast<Hittable *>(&hg);
    }
  }
  __device__ void mk_hittable_list(Hittable **&hs,
                                   curandState *loc) {
    for (int i = 0; i < nb_groups; i++) {
      SceneGroup group;
      fill_group(group, i);
      HittableGroup hg = group.to_hittable(loc);
      hs[i] = static_cast<Hittable *>(&hg);
    }
  }
  __host__ __device__ Hittables to_hittables() {
    Hittable **hs = new Hittable *[nb_groups];
    mk_hittable_list(hs);
    order_scene(hs, nb_groups);
    Hittables hits(hs, nb_groups);
    return hits;
  }
  __host__ __device__ Hittables
  to_hittables(unsigned char *&td) {
    Hittable **hs = new Hittable *[nb_groups];
    mk_hittable_list(hs, td);
    order_scene(hs, nb_groups);
    Hittables hits(hs, nb_groups);
    return hits;
  }
  __device__ Hittables to_hittables(unsigned char *&td,
                                    curandState *loc) {
    Hittable **hs = new Hittable *[nb_groups];
    mk_hittable_list(hs, td, loc);
    order_scene(hs, nb_groups);
    Hittables hits(hs, nb_groups);
    return hits;
  }
  __device__ Hittables to_hittables(curandState *loc) {
    Hittable **hs = new Hittable *[nb_groups];
    mk_hittable_list(hs, loc);
    order_scene(hs, nb_groups);
    Hittables hits(hs, nb_groups);
    return hits;
  }
};
#endif
