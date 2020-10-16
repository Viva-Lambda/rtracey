#pragma once
#include <debug.hpp>
#include <group.cuh>
#include <groupparam.cuh>
#include <ray.cuh>
#include <scenegroupparam.cuh>
#include <sceneprim.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct SceneObjects {
  // texture params
  int *ttypes;
  float *tp1xs, *tp1ys, *tp1zs;
  float *scales;

  // image params
  int *widths, *heights, *bytes_per_pixels, *image_indices;
  unsigned char *tdata;
  int tsize;

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
  int *gtypes;
  int *group_starts, *group_sizes, *group_ids;
  float *g_densities;
  // group texture params
  int *g_ttypes;
  float *g_tp1xs, *g_tp1ys, *g_tp1zs;
  float *g_scales;
  int *g_widths, *g_heights, *g_bpps, *g_indices;

  int nb_groups;
  int nb_prims;

  curandState *rand;

  __host__ __device__ SceneObjects()
      : tdata(nullptr), tsize(0), rand(nullptr) {}
  __host__ __device__ SceneObjects(GroupParam *gs, int nb_g)
      : nb_groups(nb_g), tdata(nullptr), tsize(0),
        rand(nullptr) {
    alloc_group_params(nb_g);
    nb_prims = get_prim_nb(gs);
    alloc_prim_params(nb_prims);
    set_groups(gs);
  }
  __device__ SceneObjects(GroupParam *gs, int nb_g,
                          curandState *r)
      : nb_groups(nb_g), tdata(nullptr), tsize(0), rand(r) {
    alloc_group_params(nb_g);
    nb_prims = get_prim_nb(gs);
    alloc_prim_params(nb_prims);
    set_groups(gs);
  }
  __device__ SceneObjects(GroupParam *gs, int nb_g,
                          unsigned char *td, int ts,
                          curandState *r)
      : nb_groups(nb_g), tdata(td), tsize(ts), rand(r) {
    alloc_group_params(nb_g);
    nb_prims = get_prim_nb(gs);
    alloc_prim_params(nb_prims);
    set_groups(gs);
  }
  __host__ __device__ SceneObjects(GroupParam *gs, int nb_g,
                                   unsigned char *td,
                                   int ts)
      : nb_groups(nb_g), tdata(td), tsize(ts),
        rand(nullptr) {
    alloc_group_params(nb_g);
    nb_prims = get_prim_nb(gs);
    alloc_prim_params(nb_prims);
    set_groups(gs);
  }
  __host__ __device__ void set_groups(GroupParam *gs) {
    for (int i = 0; i < nb_groups; i++) {
      GroupParam g = gs[i];
      group_sizes[i] = (*g.group_size);
      group_ids[i] = (*g.group_id);
      gtypes[i] = *g.gtype;
      int gstart = i * group_sizes[i];
      // group_starts[i] = gstart == 0 ? g.group_size :
      // gstart;
      group_starts[i] = gstart;

      for (int j = 0; j < (*g.group_size); j++) {
        int gindex = gstart + j;
        Primitive prim = g.get(j);
        set_primitive(prim, gindex);
      }
    }
  }

  __host__ __device__ int get_prim_nb(GroupParam *gs) {
    int nb_prim = 0;
    for (int i = 0; i < nb_groups; i++) {
      GroupParam g = gs[i];
      nb_prim += (*g.group_size);
    }
    return nb_prim;
  }
  __device__ void set_rand(curandState *loc) { rand = loc; }
  __host__ __device__ void
  set_group_texture(const GroupParam &g, int i) {
    g_densities[i] = *g.density;
    g_ttypes[i] = *g.tparam.ttype;
    g_tp1xs[i] = *g.tparam.tp1x;
    g_tp1ys[i] = *g.tparam.tp1y;
    g_tp1zs[i] = *g.tparam.tp1z;
    g_scales[i] = *g.tparam.scale;
    g_widths[i] = *g.tparam.width;
    g_heights[i] = *g.tparam.height;
    g_bpps[i] = *g.tparam.bytes_per_pixel;
    g_indices[i] = *g.tparam.index;
  }
  __host__ __device__ void set_primitive(Primitive &p,
                                         int gindex) {
    ttypes[gindex] = *p.mparam.tparam.ttype;

    tp1xs[gindex] = *p.mparam.tparam.tp1x;
    tp1ys[gindex] = *p.mparam.tparam.tp1y;
    tp1zs[gindex] = *p.mparam.tparam.tp1z;
    scales[gindex] = *p.mparam.tparam.scale;
    widths[gindex] = *p.mparam.tparam.width;
    heights[gindex] = *p.mparam.tparam.height;
    bytes_per_pixels[gindex] =
        *p.mparam.tparam.bytes_per_pixel;
    image_indices[gindex] = *p.mparam.tparam.index;
    mtypes[gindex] = *p.mparam.mtype;
    fuzz_ref_idxs[gindex] = *p.mparam.fuzz_ref_idx;
    htypes[gindex] = *p.hparam.htype;

    p1xs[gindex] = *p.hparam.p1x;
    p1ys[gindex] = *p.hparam.p1y;
    p1zs[gindex] = *p.hparam.p1z;

    p2xs[gindex] = *p.hparam.p2x;
    p2ys[gindex] = *p.hparam.p2y;
    p2zs[gindex] = *p.hparam.p2z;

    n1xs[gindex] = *p.hparam.n1x;
    n1ys[gindex] = *p.hparam.n1y;
    n1zs[gindex] = *p.hparam.n1z;
    rads[gindex] = *p.hparam.radius;
    prim_group_indices[gindex] = *p.group_index;
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
    tp1xs = new float[nb_ps];
    tp1ys = new float[nb_ps];
    tp1zs = new float[nb_ps];
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
    const float *gtp1x = (const float *)&g_tp1xs[gindex];
    const float *gtp1y = (const float *)&g_tp1ys[gindex];
    const float *gtp1z = (const float *)&g_tp1zs[gindex];
    const int *gwidth = (const int *)&g_widths[gindex];
    const int *gh = (const int *)&g_heights[gindex];
    const int *gbpp = (const int *)&g_bpps[gindex];
    const int *g_ind = (const int *)&g_indices[gindex];
    const ImageParam img_p =
        ImageParam(gwidth, gh, gbpp, g_ind);
    const int *gttype = (const int *)&g_ttypes[gindex];
    const float *gscl = (const float *)&g_scales[gindex];
    TextureParam tpr(gttype, gtp1x, gtp1y, gtp1z, gscl,
                     img_p);
    return tpr;
  }
  __device__ TextureParam get_group_texture_param(
      int gindex, curandState *loc) const {
    const float *gtp1x = (const float *)&g_tp1xs[gindex];
    const float *gtp1y = (const float *)&g_tp1ys[gindex];
    const float *gtp1z = (const float *)&g_tp1zs[gindex];

    const int *gwidth = &g_widths[gindex];
    const int *gh = &g_heights[gindex];
    const int *gbpp = &g_bpps[gindex];
    const int *g_ind = &g_indices[gindex];
    const ImageParam img_p =
        ImageParam(gwidth, gh, gbpp, g_ind);
    const int *gttype = &g_ttypes[gindex];
    const float *gscl = &g_scales[gindex];

    TextureParam tpr(loc, gttype, gtp1x, gtp1y, gtp1z, gscl,
                     img_p);
    return tpr;
  }

  // __device__ void set_curand(curandState *lc) { loc = lc;
  // }
  __host__ SceneObjects to_device_thrust() {
    SceneObjects sobjs;
    sobjs.nb_prims = nb_prims;
    sobjs.nb_groups = nb_groups;

    thrust::device_ptr<int> d_ttypes;
    upload_thrust<int>(d_ttypes, ttypes, nb_prims);
    sobjs.ttypes = thrust::raw_pointer_cast(d_ttypes);

    thrust::device_ptr<float> d_tp1xs;
    upload_thrust<float>(d_tp1xs, tp1xs, nb_prims);
    sobjs.tp1xs = thrust::raw_pointer_cast(d_tp1xs);

    thrust::device_ptr<float> d_tp1ys;
    upload_thrust<float>(d_tp1ys, tp1ys, nb_prims);
    sobjs.tp1ys = thrust::raw_pointer_cast(d_tp1ys);

    thrust::device_ptr<float> d_tp1zs;
    upload_thrust<float>(d_tp1zs, tp1zs, nb_prims);
    sobjs.tp1zs = thrust::raw_pointer_cast(d_tp1zs);

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

    thrust::device_ptr<unsigned char> d_tdata;
    upload_thrust<unsigned char>(d_tdata, tdata, tsize);
    sobjs.tdata = thrust::raw_pointer_cast(d_tdata);

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

    thrust::device_ptr<int> d_gtypes;
    upload_thrust<int>(d_gtypes, gtypes, nb_groups);
    sobjs.gtypes = thrust::raw_pointer_cast(d_gtypes);

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
  __host__ SceneObjects to_device() {
    SceneObjects sobjs;
    sobjs.nb_prims = nb_prims;
    sobjs.nb_groups = nb_groups;

    cudaError_t err;
    int *d_ttypes;
    sobjs.ttypes =
        upload<int>(d_ttypes, ttypes, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_tp1xs;
    sobjs.tp1xs =
        upload<float>(d_tp1xs, tp1xs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_tp1ys;
    sobjs.tp1ys =
        upload<float>(d_tp1ys, tp1ys, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_tp1zs;
    sobjs.tp1zs =
        upload<float>(d_tp1zs, tp1zs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_scales;
    sobjs.scales =
        upload<float>(d_scales, scales, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_widths;
    sobjs.widths =
        upload<int>(d_widths, widths, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_heights;
    sobjs.heights =
        upload<int>(d_heights, heights, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_bpps;
    sobjs.bytes_per_pixels = upload<int>(
        d_bpps, bytes_per_pixels, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_img_indices;
    sobjs.image_indices = upload<int>(
        d_img_indices, image_indices, nb_prims, err);
    CUDA_CONTROL(err);

    unsigned char *d_tdata;
    sobjs.tdata =
        upload<unsigned char>(d_tdata, tdata, tsize, err);
    CUDA_CONTROL(err);

    int *d_mtypes;
    sobjs.mtypes =
        upload<int>(d_mtypes, mtypes, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_fuzzs;
    sobjs.fuzz_ref_idxs = upload<float>(
        d_fuzzs, fuzz_ref_idxs, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_htypes;
    sobjs.htypes =
        upload<int>(d_htypes, htypes, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p1xs;
    sobjs.p1xs = upload<float>(d_p1xs, p1xs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p1ys;
    sobjs.p1ys = upload<float>(d_p1ys, p1ys, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p1zs;
    sobjs.p1zs = upload<float>(d_p1zs, p1zs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p2xs;
    sobjs.p2xs = upload<float>(d_p2xs, p2xs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p2ys;
    sobjs.p2ys = upload<float>(d_p2ys, p2ys, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p2zs;
    sobjs.p2zs = upload<float>(d_p2zs, p2zs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_n1xs;
    sobjs.n1xs = upload<float>(d_n1xs, n1xs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_n1ys;
    sobjs.n1ys = upload<float>(d_n1ys, n1ys, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_n1zs;
    sobjs.n1zs = upload<float>(d_n1zs, n1zs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_rads;
    sobjs.rads = upload<float>(d_rads, rads, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_prim_g_indices;
    sobjs.prim_group_indices =
        upload<int>(d_prim_g_indices, prim_group_indices,
                    nb_prims, err);
    CUDA_CONTROL(err);

    int *d_gstarts;
    sobjs.group_starts = upload<int>(
        d_gstarts, group_starts, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_gsizes;

    sobjs.group_sizes =
        upload<int>(d_gsizes, group_sizes, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_gids;
    sobjs.group_ids =
        upload<int>(d_gids, group_ids, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_gts;
    sobjs.gtypes =
        upload<int>(d_gts, gtypes, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_dens;

    sobjs.g_densities = upload<float>(d_g_dens, g_densities,
                                      nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_ttypes;
    sobjs.g_ttypes =
        upload<int>(d_g_ttypes, g_ttypes, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_tp1xs;
    sobjs.g_tp1xs =
        upload<float>(d_g_tp1xs, g_tp1xs, nb_groups, err);

    CUDA_CONTROL(err);

    float *d_g_tp1ys;
    sobjs.g_tp1ys =
        upload<float>(d_g_tp1ys, g_tp1ys, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_tp1zs;
    sobjs.g_tp1zs =
        upload<float>(d_g_tp1zs, g_tp1zs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_scales;
    sobjs.g_scales =
        upload<float>(d_g_scales, g_scales, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_widths;
    sobjs.g_widths =
        upload<int>(d_g_widths, g_widths, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_heights;
    sobjs.g_heights =
        upload<int>(d_g_heights, g_heights, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_bpps;
    sobjs.g_bpps =
        upload<int>(d_g_bpps, g_bpps, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_indices;

    sobjs.g_indices =
        upload<int>(d_g_indices, g_indices, nb_groups, err);
    CUDA_CONTROL(err);

    return sobjs;
  }
  __host__ SceneObjects to_host() {
    SceneObjects sobjs;
    sobjs.nb_prims = nb_prims;
    sobjs.nb_groups = nb_groups;

    cudaError_t err;
    int *d_ttypes;
    sobjs.ttypes =
        download<int>(d_ttypes, ttypes, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_tp1xs;
    sobjs.tp1xs =
        download<float>(d_tp1xs, tp1xs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_tp1ys;
    sobjs.tp1ys =
        download<float>(d_tp1ys, tp1ys, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_tp1zs;
    sobjs.tp1zs =
        download<float>(d_tp1zs, tp1zs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_scales;
    sobjs.scales =
        download<float>(d_scales, scales, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_widths;
    sobjs.widths =
        download<int>(d_widths, widths, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_heights;
    sobjs.heights =
        download<int>(d_heights, heights, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_bpps;
    sobjs.bytes_per_pixels = download<int>(
        d_bpps, bytes_per_pixels, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_img_indices;
    sobjs.image_indices = download<int>(
        d_img_indices, image_indices, nb_prims, err);
    CUDA_CONTROL(err);

    unsigned char *d_tdata;
    sobjs.tdata =
        download<unsigned char>(d_tdata, tdata, tsize, err);
    CUDA_CONTROL(err);

    int *d_mtypes;
    sobjs.mtypes =
        download<int>(d_mtypes, mtypes, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_fuzzs;
    sobjs.fuzz_ref_idxs = download<float>(
        d_fuzzs, fuzz_ref_idxs, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_htypes;
    sobjs.htypes =
        download<int>(d_htypes, htypes, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p1xs;
    sobjs.p1xs =
        download<float>(d_p1xs, p1xs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p1ys;
    sobjs.p1ys =
        download<float>(d_p1ys, p1ys, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p1zs;
    sobjs.p1zs =
        download<float>(d_p1zs, p1zs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p2xs;
    sobjs.p2xs =
        download<float>(d_p2xs, p2xs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p2ys;
    sobjs.p2ys =
        download<float>(d_p2ys, p2ys, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_p2zs;
    sobjs.p2zs =
        download<float>(d_p2zs, p2zs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_n1xs;
    sobjs.n1xs =
        download<float>(d_n1xs, n1xs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_n1ys;
    sobjs.n1ys =
        download<float>(d_n1ys, n1ys, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_n1zs;
    sobjs.n1zs =
        download<float>(d_n1zs, n1zs, nb_prims, err);
    CUDA_CONTROL(err);

    float *d_rads;
    sobjs.rads =
        download<float>(d_rads, rads, nb_prims, err);
    CUDA_CONTROL(err);

    int *d_prim_g_indices;
    sobjs.prim_group_indices =
        download<int>(d_prim_g_indices, prim_group_indices,
                      nb_prims, err);
    CUDA_CONTROL(err);

    int *d_gstarts;
    sobjs.group_starts = download<int>(
        d_gstarts, group_starts, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_gsizes;

    sobjs.group_sizes = download<int>(d_gsizes, group_sizes,
                                      nb_groups, err);
    CUDA_CONTROL(err);

    int *d_gids;
    sobjs.group_ids =
        download<int>(d_gids, group_ids, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_gts;
    sobjs.gtypes =
        download<int>(d_gts, gtypes, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_dens;

    sobjs.g_densities = download<float>(
        d_g_dens, g_densities, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_ttypes;
    sobjs.g_ttypes =
        download<int>(d_g_ttypes, g_ttypes, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_tp1xs;
    sobjs.g_tp1xs =
        download<float>(d_g_tp1xs, g_tp1xs, nb_groups, err);

    CUDA_CONTROL(err);

    float *d_g_tp1ys;
    sobjs.g_tp1ys =
        download<float>(d_g_tp1ys, g_tp1ys, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_tp1zs;
    sobjs.g_tp1zs =
        download<float>(d_g_tp1zs, g_tp1zs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_scales;
    sobjs.g_scales = download<float>(d_g_scales, g_scales,
                                     nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_widths;
    sobjs.g_widths =
        download<int>(d_g_widths, g_widths, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_heights;
    sobjs.g_heights = download<int>(d_g_heights, g_heights,
                                    nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_bpps;
    sobjs.g_bpps =
        download<int>(d_g_bpps, g_bpps, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_indices;

    sobjs.g_indices = download<int>(d_g_indices, g_indices,
                                    nb_groups, err);
    CUDA_CONTROL(err);

    return sobjs;
  }
  __host__ void d_free() {
    cudaFree(ttypes);
    cudaFree(tp1xs);
    cudaFree(tp1ys);
    cudaFree(tp1zs);
    cudaFree(scales);
    cudaFree(widths);
    cudaFree(heights);
    cudaFree(bytes_per_pixels);
    cudaFree(image_indices);
    cudaFree(tdata);
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
  __host__ void h_free() {
    delete[] ttypes;
    delete[] tp1xs;
    delete[] tp1ys;
    delete[] tp1zs;
    delete[] scales;
    delete[] widths;
    delete[] heights;
    delete[] bytes_per_pixels;
    delete[] image_indices;
    delete[] tdata;
    delete[] mtypes;
    delete[] fuzz_ref_idxs;
    delete[] htypes;

    delete[] p1xs;
    delete[] p1ys;
    delete[] p1zs;

    delete[] p2xs;
    delete[] p2ys;
    delete[] p2zs;

    delete[] n1xs;
    delete[] n1ys;
    delete[] n1zs;

    delete[] rads;
    delete[] prim_group_indices;

    delete[] group_starts;
    delete[] group_sizes;
    delete[] group_ids;
    delete[] gtypes;
    delete[] g_densities;
    delete[] g_ttypes;
    delete[] g_tp1xs;
    delete[] g_tp1ys;
    delete[] g_tp1zs;
    delete[] g_scales;
    delete[] g_widths;
    delete[] g_heights;
    delete[] g_bpps;
    delete[] g_indices;
  }
  __host__ __device__ HittableParam
  get_hparam(int gstart, int group_id) const {
    const int *htype = (const int *)&htypes[gstart];
    const float *p1x = (const float *)&p1xs[gstart];
    const float *p1y = (const float *)&p1ys[gstart];
    const float *p1z = (const float *)&p1zs[gstart];
    const float *p2x = (const float *)&p2xs[gstart];
    const float *p2y = (const float *)&p2ys[gstart];
    const float *p2z = (const float *)&p2zs[gstart];
    const float *n1x = (const float *)&n1xs[gstart];
    const float *n1y = (const float *)&n1ys[gstart];
    const float *n1z = (const float *)&n1zs[gstart];
    const float *radius = (const float *)&rads[gstart];

    HittableParam ht(htype, p1x, p1y, p1z, p2x, p2y, p2z,
                     n1x, n1y, n1z, radius);
    return ht;
  }
  __host__ __device__ ImageParam
  get_image_param(int gstart, int group_id) const {
    const int *w = (const int *)&widths[gstart];
    const int *h = (const int *)&heights[gstart];
    const int *bpp = (const int *)&bytes_per_pixels[gstart];
    const int *idx = (const int *)&image_indices[gstart];
    ImageParam imp(w, h, bpp, idx);
    return imp;
  }
  __host__ __device__ TextureParam
  get_texture_param(int gstart, int group_id,
                    const ImageParam &imp) const {
    const float *tp1x = (const float *)&tp1xs[gstart];
    const float *tp1y = (const float *)&tp1ys[gstart];
    const float *tp1z = (const float *)&tp1zs[gstart];
    const int *ttype = (const int *)&ttypes[gstart];
    const float *sc = (const float *)&scales[gstart];
    TextureParam tp(ttype, tp1x, tp1y, tp1z, sc, imp);
    return tp;
  }
  __device__ TextureParam get_texture_param(
      int gstart, int group_id, const ImageParam &imp,
      curandState *loc) const {
    const float *tp1x = (const float *)&tp1xs[gstart];
    const float *tp1y = (const float *)&tp1ys[gstart];
    const float *tp1z = (const float *)&tp1zs[gstart];
    const int *ttype = (const int *)&ttypes[gstart];
    const float *sc = (const float *)&scales[gstart];
    TextureParam tp(loc, ttype, tp1x, tp1y, tp1z, sc, imp);
    return tp;
  }
  __host__ __device__ MaterialParam
  get_material_param(int gstart, int group_id,
                     const TextureParam &tp) const {
    const int *mt = (const int *)&mtypes[gstart];
    const float *fzz =
        (const float *)&fuzz_ref_idxs[gstart];
    MaterialParam mp(tp, mt, fzz);
    return mp;
  }
  __host__ __device__ MaterialParam
  get_material_param(int gstart, int group_id) const {
    const ImageParam imp =
        get_image_param(gstart, group_id);
    const TextureParam tp =
        get_texture_param(gstart, group_id, imp);
    const int *mt = (const int *)&mtypes[gstart];
    const float *fzz =
        (const float *)&fuzz_ref_idxs[gstart];

    MaterialParam mp(tp, mt, fzz);
    return mp;
  }
  __device__ MaterialParam get_material_param(
      int gstart, int group_id, curandState *loc) const {
    const ImageParam imp =
        get_image_param(gstart, group_id);
    const TextureParam tp =
        get_texture_param(gstart, group_id, imp, loc);
    const int *mt = (const int *)&mtypes[gstart];
    const float *fzz =
        (const float *)&fuzz_ref_idxs[gstart];

    MaterialParam mp(tp, mt, fzz);
    return mp;
  }
  __host__ __device__ Primitive
  get_primitive(int gstart, int group_id) const {
    const HittableParam ht = get_hparam(gstart, group_id);
    const ImageParam imp =
        get_image_param(gstart, group_id);
    const TextureParam tp =
        get_texture_param(gstart, group_id, imp);
    const MaterialParam mp =
        get_material_param(gstart, group_id, tp);
    const int *gindex =
        (const int *)&prim_group_indices[gstart];
    const int *gid = (const int *)&group_id;
    Primitive prim = Primitive(mp, ht, gindex, gid);
    return prim;
  }
  __device__ Primitive get_primitive(
      int gstart, int group_id, curandState *loc) const {
    const HittableParam ht = get_hparam(gstart, group_id);
    const ImageParam imp =
        get_image_param(gstart, group_id);
    const TextureParam tp =
        get_texture_param(gstart, group_id, imp, loc);
    const MaterialParam mp =
        get_material_param(gstart, group_id, tp);
    const int *gindex =
        (const int *)&prim_group_indices[gstart];
    const int *gid = (const int *)&group_id;

    Primitive prim = Primitive(mp, ht, gindex, gid);
    return prim;
  }
  __device__ bool group_hit(int gindex, const Ray &r,
                            float d_min, float d_max,
                            HitRecord &rec) const {
    const int *ggtype = (const int *)&gtypes[gindex];
    const int *gsize = (const int *)&group_sizes[gindex];
    const int *g_id = (const int *)&group_ids[gindex];
    const float *g_dens =
        (const float *)&g_densities[gindex];
    const TextureParam tparam =
        get_group_texture_param(gindex, loc);

    const int *gstart = (const int *)&group_starts[gindex];
    bool any_hit = false;
    for (int i = 0; i < (*gsize); i++) {
      int g_start = (*gstart) + i;
      const HittableParam ht = get_hparam(gstart, group_id);
      const ImageParam imp =
          get_image_param(gstart, group_id);
      const TextureParam tp =
          get_texture_param(gstart, group_id, imp);
      const MaterialParam mp =
          get_material_param(gstart, group_id, tp);
      const int *gindex =
          (const int *)&prim_group_indices[gstart];
      const int *gid = (const int *)&group_id;
      Primitive p(mp, ht, gindex, gid);
      any_hit = SceneHittable<Primitive>::hit(p, r, d_min,
                                              d_max, rec);
    }
    return any_hit;
  }
  __device__ bool d_hit(const Ray &r, float d_min,
                        float d_max, HitRecord &rec) const {
    HitRecord temp;
    bool hit_anything = false;
    float closest_far = d_max;
    for (int i = 0; i < nb_groups; i++) {
      GroupParam g = fill_group(i, rand);
      int gstart = group_starts[i];
      bool is_hit = SceneHittable<GroupParam>::hit(
          g, r, d_min, closest_far, temp);
      if (is_hit == true) {
        hit_anything = is_hit;
        closest_far = temp.t;
        rec = temp;
        rec.primitive_index = gstart + temp.group_index;
        rec.group_id = group_ids[i];
      }
    }
    return hit_anything;
  }
  __device__ bool h_hit(const Ray &r, float d_min,
                        float d_max, HitRecord &rec) const {
    HitRecord temp;
    bool hit_anything = false;
    float closest_far = d_max;
    for (int i = 0; i < nb_groups; i++) {
      bool is_hit = group_hit(i, r, d_min, d_max, rec);
      if (is_hit == true) {
        hit_anything = is_hit;
        closest_far = temp.t;
        rec = temp;
        rec.primitive_index = gstart + rec.group_index;
        rec.group_id = group_ids[i];
      }
    }
    return hit_anything;
  }
  __device__ bool hit(const Ray &r, float d_min,
                      float d_max, HitRecord &rec) const {
    if (rand != nullptr) {
      return d_hit(r, d_min, d_max, rec);
    } else {
      return h_hit(r, d_min, d_max, rec);
    }
  }
  __device__ bool scatter(const Ray &r,
                          const HitRecord &rec,
                          Color &attenuation, Ray &r_out,
                          float &pdf_v) const {
    int prim_index = rec.primitive_index;
    int group_id = rec.group_id;
    MaterialParam prim =
        get_material_param(prim_index, group_id, rand);
    return SceneMaterial<MaterialParam>::scatter(
        prim, r, rec, attenuation, r_out, pdf_v, rand);
  }
  __device__ Color emitted(const HitRecord &rec) const {
    int prim_index = rec.primitive_index;
    int group_id = rec.group_id;
    MaterialParam prim =
        get_material_param(prim_index, group_id, rand);
    return SceneMaterial<MaterialParam>::emitted(
        prim, rec.u, rec.v, rec.p);
  }
  __device__ float scattering_pdf(const HitRecord &rec,
                                  const Ray &r,
                                  const Ray &r_out) const {
    //
    int prim_index = rec.primitive_index;
    int group_id = rec.group_id;
    MaterialParam prim =
        get_material_param(prim_index, group_id, rand);
    return SceneMaterial<MaterialParam>::scattering_pdf(
        prim, r, rec, r_out);
  }
};
