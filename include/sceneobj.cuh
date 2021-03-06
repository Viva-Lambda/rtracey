#pragma once
#include <debug.hpp>
#include <group.cuh>
#include <groupparam.cuh>
#include <primitive.cuh>
#include <ray.cuh>
#include <scenetype.cuh>
#include <utils.cuh>
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
  int *g_mtypes;
  float *g_fuzz_ref_idxs;
  float *g_minxs, *g_minys, *g_minzs;
  float *g_maxxs, *g_maxys, *g_maxzs;

  // transformation params
  int *g_transtypes;
  //
  float *g_stepxs;
  float *g_stepys;
  float *g_stepzs;
  //
  float *g_degrees;
  //
  float *g_axis_xs;
  float *g_axis_ys;
  float *g_axis_zs;
  //
  float *g_tscales;

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
  __device__ SceneObjects(GroupParam *&gs, int nb_g,
                          unsigned char *td, int ts,
                          curandState *r)
      : nb_groups(nb_g), tsize(ts), rand(r) {
    alloc_group_params(nb_g);
    nb_prims = get_prim_nb(gs);
    alloc_prim_params(nb_prims);
    set_groups(gs);
    deepcopy(tdata, td, tsize);
  }
  __host__ __device__ SceneObjects(GroupParam *&gs,
                                   int nb_g,
                                   unsigned char *td,
                                   int ts)
      : nb_groups(nb_g), tsize(ts), rand(nullptr) {
    alloc_group_params(nb_g);
    nb_prims = get_prim_nb(gs);
    alloc_prim_params(nb_prims);
    set_groups(gs);
    deepcopy(tdata, td, tsize);
  }
  __host__ __device__ void set_groups(GroupParam *gs) {
    int gcount = 0;
    for (int i = 0; i < nb_groups; i++) {
      GroupParam g = gs[i];
      group_sizes[i] = g.group_size;
      group_ids[i] = g.group_id;
      gtypes[i] = g.gtype;
      int gstart = gcount;
      gcount += group_sizes[i];
      // group_starts[i] = gstart == 0 ? g.group_size :
      // gstart;
      group_starts[i] = gstart;
      set_group_texture(g, i);

      for (int j = 0; j < g.group_size; j++) {
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
      nb_prim += g.group_size;
    }
    return nb_prim;
  }
  __device__ void set_rand(curandState *loc) { rand = loc; }
  __host__ __device__ void
  set_group_texture(const GroupParam &g, int i) {
    g_densities[i] = g.density;
    g_ttypes[i] = g.ttype;
    g_tp1xs[i] = g.tp1x;
    g_tp1ys[i] = g.tp1y;
    g_tp1zs[i] = g.tp1z;
    g_scales[i] = g.scale;
    g_widths[i] = g.width;
    g_heights[i] = g.height;
    g_bpps[i] = g.bytes_per_pixel;
    g_indices[i] = g.index;
    g_mtypes[i] = g.mtype;
    g_fuzz_ref_idxs[i] = g.fuzz_ref_idx;
    //
    g_minxs[i] = g.minx;
    g_minys[i] = g.miny;
    g_minzs[i] = g.minz;
    //
    g_maxxs[i] = g.maxx;
    g_maxys[i] = g.maxy;
    g_maxzs[i] = g.maxz;
    //
    g_transtypes[i] = g.transtype;
    g_stepxs[i] = g.stepx;
    g_stepys[i] = g.stepy;
    g_stepzs[i] = g.stepz;

    g_degrees[i] = g.degree;
    //
    g_axis_xs[i] = g.axis_x;
    g_axis_ys[i] = g.axis_y;
    g_axis_zs[i] = g.axis_z;
    g_tscales[i] = g.tscale;
  }
  __host__ __device__ void set_primitive(const Primitive &p,
                                         int gindex) {
    ttypes[gindex] = p.mparam.tparam.ttype;

    tp1xs[gindex] = p.mparam.tparam.tp1x;
    tp1ys[gindex] = p.mparam.tparam.tp1y;
    tp1zs[gindex] = p.mparam.tparam.tp1z;

    scales[gindex] = p.mparam.tparam.scale;
    widths[gindex] = p.mparam.tparam.width;
    heights[gindex] = p.mparam.tparam.height;
    bytes_per_pixels[gindex] =
        p.mparam.tparam.bytes_per_pixel;
    image_indices[gindex] = p.mparam.tparam.index;
    mtypes[gindex] = p.mparam.mtype;
    fuzz_ref_idxs[gindex] = p.mparam.fuzz_ref_idx;
    htypes[gindex] = p.htype;

    p1xs[gindex] = p.p1x;
    p1ys[gindex] = p.p1y;
    p1zs[gindex] = p.p1z;

    p2xs[gindex] = p.p2x;
    p2ys[gindex] = p.p2y;
    p2zs[gindex] = p.p2z;

    n1xs[gindex] = p.n1x;
    n1ys[gindex] = p.n1y;
    n1zs[gindex] = p.n1z;
    rads[gindex] = p.radius;
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
    g_mtypes = new int[nb_g];
    g_fuzz_ref_idxs = new float[nb_g];

    g_minxs = new float[nb_g];
    g_minys = new float[nb_g];
    g_minzs = new float[nb_g];

    g_maxxs = new float[nb_g];
    g_maxys = new float[nb_g];
    g_maxzs = new float[nb_g];

    //
    g_transtypes = new int[nb_g];
    g_stepxs = new float[nb_g];
    g_stepys = new float[nb_g];
    g_stepzs = new float[nb_g];
    g_degrees = new float[nb_g];
    //
    g_axis_xs = new float[nb_g];
    g_axis_ys = new float[nb_g];
    g_axis_zs = new float[nb_g];
    //
    g_tscales = new float[nb_g];
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
  __host__ SceneObjects to_device_thrust() {
    SceneObjects sobjs;
    sobjs.nb_prims = nb_prims;
    sobjs.nb_groups = nb_groups;

    to_device_g_prim_tex(sobjs);
    to_device_g_hittable(sobjs);

    thrust::device_ptr<unsigned char> d_tdata;
    upload_thrust<unsigned char>(d_tdata, tdata, tsize);
    sobjs.tdata = thrust::raw_pointer_cast(d_tdata);

    thrust::device_ptr<int> d_mtypes;
    upload_thrust<int>(d_mtypes, mtypes, nb_prims);
    sobjs.mtypes = thrust::raw_pointer_cast(d_mtypes);

    thrust::device_ptr<float> d_fuzzs;
    upload_thrust<float>(d_fuzzs, fuzz_ref_idxs, nb_prims);
    sobjs.fuzz_ref_idxs = thrust::raw_pointer_cast(d_fuzzs);

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

    to_device_g_img(sobjs);

    to_device_g_texcolor(sobjs);

    thrust::device_ptr<float> d_g_scales;
    upload_thrust<float>(d_g_scales, g_scales, nb_groups);
    sobjs.g_scales = thrust::raw_pointer_cast(d_g_scales);
    thrust::device_ptr<int> d_g_mtypes;
    upload_thrust<int>(d_g_mtypes, g_mtypes, nb_groups);
    sobjs.g_mtypes = thrust::raw_pointer_cast(d_g_mtypes);

    thrust::device_ptr<float> d_g_fuzz_ref_idxs;
    upload_thrust<float>(d_g_fuzz_ref_idxs, g_fuzz_ref_idxs,
                         nb_groups);
    sobjs.g_fuzz_ref_idxs =
        thrust::raw_pointer_cast(d_g_fuzz_ref_idxs);

    to_device_g_minmax(sobjs);
    //
    to_device_transparam_thrust(sobjs);

    return sobjs;
  }
  __host__ void to_device_g_prim_tex(SceneObjects &sobjs) {
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
  }
  __host__ void to_device_g_hittable(SceneObjects &sobjs) {
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
  }
  __host__ void to_device_g_img(SceneObjects &sobjs) {
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
  }
  __host__ void to_device_g_texcolor(SceneObjects &sobjs) {
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
  }
  __host__ void to_device_g_minmax(SceneObjects &sobjs) {
    thrust::device_ptr<float> d_g_minxs;
    upload_thrust<float>(d_g_minxs, g_minxs, nb_groups);
    sobjs.g_minxs = thrust::raw_pointer_cast(d_g_minxs);

    thrust::device_ptr<float> d_g_minys;
    upload_thrust<float>(d_g_minys, g_minys, nb_groups);
    sobjs.g_minys = thrust::raw_pointer_cast(d_g_minys);

    thrust::device_ptr<float> d_g_minzs;
    upload_thrust<float>(d_g_minzs, g_minzs, nb_groups);
    sobjs.g_minzs = thrust::raw_pointer_cast(d_g_minzs);

    thrust::device_ptr<float> d_g_maxxs;
    upload_thrust<float>(d_g_maxxs, g_maxxs, nb_groups);
    sobjs.g_maxxs = thrust::raw_pointer_cast(d_g_maxxs);

    thrust::device_ptr<float> d_g_maxys;
    upload_thrust<float>(d_g_maxys, g_maxys, nb_groups);
    sobjs.g_maxys = thrust::raw_pointer_cast(d_g_maxys);

    thrust::device_ptr<float> d_g_maxzs;
    upload_thrust<float>(d_g_maxzs, g_maxzs, nb_groups);
    sobjs.g_maxzs = thrust::raw_pointer_cast(d_g_maxzs);
  }
  __host__ SceneObjects to_device() {
    SceneObjects sobjs;
    sobjs.nb_prims = nb_prims;
    sobjs.nb_groups = nb_groups;

    to_d_g_texcolor(sobjs);
    to_d_g_hittable(sobjs);

    cudaError_t err;
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

    to_d_g_g_texcolor(sobjs);

    int *d_g_mtypes;
    sobjs.g_mtypes =
        upload<int>(d_g_mtypes, g_mtypes, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_fuzz_ref_idxs;
    sobjs.g_fuzz_ref_idxs = upload<float>(
        d_g_fuzz_ref_idxs, g_fuzz_ref_idxs, nb_groups, err);
    CUDA_CONTROL(err);

    to_d_g_minmax(sobjs);
    //
    to_device_transparam(sobjs);

    return sobjs;
  }
  __host__ void to_device_transparam(SceneObjects &sobjs) {
    //
    cudaError_t err;
    int *d_g_trans_types;
    sobjs.g_transtypes = upload<int>(
        d_g_trans_types, g_transtypes, nb_groups, err);
    CUDA_CONTROL(err);
    //
    float *d_g_stepxs;
    sobjs.g_stepxs =
        upload<float>(d_g_stepxs, g_stepxs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_stepys;
    sobjs.g_stepys =
        upload<float>(d_g_stepys, g_stepys, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_stepzs;
    sobjs.g_stepzs =
        upload<float>(d_g_stepzs, g_stepzs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_degrees;
    sobjs.g_degrees = upload<float>(d_g_degrees, g_degrees,
                                    nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_axis_xs;
    sobjs.g_axis_xs = upload<float>(d_g_axis_xs, g_axis_xs,
                                    nb_groups, err);
    CUDA_CONTROL(err);
    float *d_g_axis_ys;
    sobjs.g_axis_ys = upload<float>(d_g_axis_ys, g_axis_ys,
                                    nb_groups, err);
    CUDA_CONTROL(err);
    float *d_g_axis_zs;
    sobjs.g_axis_zs = upload<float>(d_g_axis_zs, g_axis_zs,
                                    nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_tscales;
    sobjs.g_tscales = upload<float>(d_g_tscales, g_tscales,
                                    nb_groups, err);
    CUDA_CONTROL(err);
  }
  __host__ void
  to_device_transparam_thrust(SceneObjects &sobjs) {

    thrust::device_ptr<int> d_g_trans_types;
    upload_thrust<int>(d_g_trans_types, g_transtypes,
                       nb_groups);
    sobjs.g_transtypes =
        thrust::raw_pointer_cast(d_g_trans_types);

    //
    thrust::device_ptr<float> d_g_stepxs;
    upload_thrust<float>(d_g_stepxs, g_stepxs, nb_groups);
    sobjs.g_stepxs = thrust::raw_pointer_cast(d_g_stepxs);

    thrust::device_ptr<float> d_g_stepys;
    upload_thrust<float>(d_g_stepys, g_stepys, nb_groups);
    sobjs.g_stepys = thrust::raw_pointer_cast(d_g_stepys);

    thrust::device_ptr<float> d_g_stepzs;
    upload_thrust<float>(d_g_stepzs, g_stepzs, nb_groups);
    sobjs.g_stepzs = thrust::raw_pointer_cast(d_g_stepzs);

    thrust::device_ptr<float> d_g_degrees;
    upload_thrust<float>(d_g_degrees, g_degrees, nb_groups);
    sobjs.g_degrees = thrust::raw_pointer_cast(d_g_degrees);

    thrust::device_ptr<float> d_g_axis_xs;
    upload_thrust<float>(d_g_axis_xs, g_axis_xs, nb_groups);
    sobjs.g_axis_xs = thrust::raw_pointer_cast(d_g_axis_xs);

    thrust::device_ptr<float> d_g_axis_ys;
    upload_thrust<float>(d_g_axis_ys, g_axis_ys, nb_groups);
    sobjs.g_axis_ys = thrust::raw_pointer_cast(d_g_axis_ys);

    thrust::device_ptr<float> d_g_axis_zs;
    upload_thrust<float>(d_g_axis_zs, g_axis_zs, nb_groups);
    sobjs.g_axis_zs = thrust::raw_pointer_cast(d_g_axis_zs);

    thrust::device_ptr<float> d_g_tscales;
    upload_thrust<float>(d_g_tscales, g_tscales, nb_groups);
    sobjs.g_tscales = thrust::raw_pointer_cast(d_g_tscales);
  }
  __host__ void to_d_g_texcolor(SceneObjects &sobjs) {
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
  }
  __host__ void to_d_g_hittable(SceneObjects &sobjs) {
    cudaError_t err;
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
  }
  __host__ void to_d_g_g_texcolor(SceneObjects &sobjs) {
    cudaError_t err;
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
  }
  __host__ void to_d_g_minmax(SceneObjects &sobjs) {
    cudaError_t err;
    float *d_g_minxs;
    sobjs.g_minxs =
        upload<float>(d_g_minxs, g_minxs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_minys;
    sobjs.g_minys =
        upload<float>(d_g_minys, g_minys, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_minzs;
    sobjs.g_minzs =
        upload<float>(d_g_minzs, g_minzs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_maxxs;
    sobjs.g_maxxs =
        upload<float>(d_g_maxxs, g_maxxs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_maxys;
    sobjs.g_maxys =
        upload<float>(d_g_maxys, g_maxys, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_maxzs;
    sobjs.g_maxzs =
        upload<float>(d_g_maxzs, g_maxzs, nb_groups, err);
    CUDA_CONTROL(err);
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

    int *d_g_mtypes;
    sobjs.g_mtypes =
        download<int>(d_g_mtypes, g_mtypes, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_fuzz_ref_idxs;
    sobjs.g_fuzz_ref_idxs = download<float>(
        d_g_fuzz_ref_idxs, g_fuzz_ref_idxs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_minxs;
    sobjs.g_minxs =
        download<float>(d_g_minxs, g_minxs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_minys;
    sobjs.g_minys =
        download<float>(d_g_minys, g_minys, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_minzs;
    sobjs.g_minzs =
        download<float>(d_g_minzs, g_minzs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_maxzs;
    sobjs.g_maxzs =
        download<float>(d_g_maxzs, g_maxzs, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_maxys;
    sobjs.g_maxys =
        download<float>(d_g_maxys, g_maxys, nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_maxxs;
    sobjs.g_maxxs =
        download<float>(d_g_maxxs, g_maxxs, nb_groups, err);
    CUDA_CONTROL(err);

    int *d_g_trans_types;
    sobjs.g_transtypes = download<int>(
        d_g_trans_types, g_transtypes, nb_groups, err);
    CUDA_CONTROL(err);
    //
    float *d_g_stepxs;
    sobjs.g_stepxs = download<float>(d_g_stepxs, g_stepxs,
                                     nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_stepys;
    sobjs.g_stepys = download<float>(d_g_stepys, g_stepys,
                                     nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_stepzs;
    sobjs.g_stepzs = download<float>(d_g_stepzs, g_stepzs,
                                     nb_groups, err);
    CUDA_CONTROL(err);

    float *d_g_degrees;
    sobjs.g_degrees = download<float>(
        d_g_degrees, g_degrees, nb_groups, err);
    CUDA_CONTROL(err);

    return sobjs;
  }
  __host__ __device__ Primitive get_prim(int index,
                                         int gindex) const {
    HittableParam hp = get_hparam(index);
    MaterialParam mp = get_mparam(index);
    Primitive p(mp, hp, prim_group_indices[index],
                group_ids[gindex]);
    return p;
  }
  __host__ __device__ HittableParam
  get_hparam(int idx) const {
    HittableParam hp(static_cast<HittableType>(htypes[idx]),
                     p1xs[idx], p1ys[idx], p1zs[idx],
                     p2xs[idx], p2ys[idx], p2zs[idx],
                     n1xs[idx], n1ys[idx], n1zs[idx],
                     rads[idx]);
    return hp;
  }
  __host__ __device__ TextureParam
  get_tparam(int idx) const {
    TextureParam t(static_cast<TextureType>(ttypes[idx]),
                   tp1xs[idx], tp1ys[idx], tp1zs[idx],
                   scales[idx], widths[idx], heights[idx],
                   bytes_per_pixels[idx],
                   image_indices[idx]);
    return t;
  }
  __host__ __device__ MaterialParam
  get_mparam(int idx) const {
    TextureParam t = get_tparam(idx);
    MaterialParam mp(t,
                     static_cast<MaterialType>(mtypes[idx]),
                     fuzz_ref_idxs[idx]);
    return mp;
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
};
