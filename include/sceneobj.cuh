#pragma once
#include <debug.hpp>
#include <ray.cuh>
#include <scenegroup.cuh>
#include <sceneparam.cuh>
#include <sceneprim.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct SceneObjects {
  // texture params
  TextureType *ttypes;
  Color *cvals;
  float *scales;

  // image params
  int *widths;
  int *heights;
  int *bytes_per_pixels;
  int *image_indices;
  unsigned char *tdata;
  int tsize;

  // material params
  MaterialType *mtypes;
  float *fuzz_ref_idxs;

  // hittable params
  HittableType *htypes;
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
  GroupType *gtypes;

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
    ttypes[gindex] = p.ttype;
    cvals[gindex] = p.cval;
    scales[gindex] = p.scale;
    widths[gindex] = p.width;
    heights[gindex] = p.height;
    bytes_per_pixels[gindex] = p.bytes_per_pixel;
    image_indices[gindex] = p.image_index;
    mtypes[gindex] = p.mtype;
    fuzz_ref_idxs[gindex] = p.fuzz_ref_idx;
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
    gtypes = new GroupType[nb_g];
  }
  __host__ __device__ void alloc_prim_params(int nb_ps) {
    ttypes = new TextureType[nb_ps];
    cvals = new Color[nb_ps];
    scales = new float[nb_ps];
    widths = new int[nb_ps];
    heights = new int[nb_ps];
    bytes_per_pixels = new int[nb_ps];
    image_indices = new int[nb_ps];
    mtypes = new MaterialType[nb_ps];
    fuzz_ref_idxs = new float[nb_ps];

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
  __host__ __device__ void fill_group(SceneGroup &group,
                                      int gindex) {
    group.gtype = gtypes[gindex];
    group.group_size = group_sizes[gindex];
    group.group_id = group_ids[gindex];
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

    TextureType *d_ttypes;
    CUDA_CONTROL(upload(d_ttypes, ttypes, nb_prims));
    sobjs.ttypes = d_ttypes;

    Color *d_cvals;
    CUDA_CONTROL(upload(d_cvals, cvals, nb_prims));
    sobjs.cvals = d_cvals;

    float *d_scales;
    CUDA_CONTROL(upload(d_scales, scales, nb_prims));
    sobjs.scales = d_scales;

    int *d_widths;
    CUDA_CONTROL(upload(d_widths, widths, nb_prims));
    sobjs.widths = d_widths;

    int *d_heights;
    CUDA_CONTROL(upload(d_heights, heights, nb_prims));
    sobjs.heights = d_heights;

    int *d_bpps;
    CUDA_CONTROL(
        upload(d_bpps, bytes_per_pixels, nb_prims));
    sobjs.bytes_per_pixels = d_bpps;

    int *d_img_indices;
    CUDA_CONTROL(
        upload(d_img_indices, image_indices, nb_prims));
    sobjs.image_indices = d_img_indices;

    unsigned char *d_tdata;
    CUDA_CONTROL(upload(d_tdata, tdata, tsize));
    sobjs.tdata = d_tdata;
    sobjs.tsize = tsize;

    MaterialType *d_mtypes;
    CUDA_CONTROL(upload(d_mtypes, mtypes, nb_prims));
    sobjs.mtypes = d_mtypes;

    float *d_fuzzs;
    CUDA_CONTROL(upload(d_fuzzs, fuzz_ref_idxs, nb_prims));
    sobjs.fuzz_ref_idxs = d_fuzzs;

    HittableType *d_htypes;
    CUDA_CONTROL(upload(d_htypes, htypes, nb_prims));
    sobjs.htypes = d_htypes;

    float *d_p1xs;
    CUDA_CONTROL(upload(d_p1xs, p1xs, nb_prims));
    sobjs.p1xs = d_p1xs;

    float *d_p1ys;
    CUDA_CONTROL(upload(d_p1ys, p1ys, nb_prims));
    sobjs.p1ys = d_p1ys;

    float *d_p1zs;
    CUDA_CONTROL(upload(d_p1zs, p1zs, nb_prims));
    sobjs.p1zs = d_p1zs;

    float *d_p2xs;
    CUDA_CONTROL(upload(d_p2xs, p2xs, nb_prims));
    sobjs.p2xs = d_p2xs;

    float *d_p2ys;
    CUDA_CONTROL(upload(d_p2ys, p2ys, nb_prims));
    sobjs.p2ys = d_p2ys;

    float *d_p2zs;
    CUDA_CONTROL(upload(d_p2zs, p2zs, nb_prims));
    sobjs.p2zs = d_p2zs;

    float *d_n1xs;
    CUDA_CONTROL(upload(d_n1xs, n1xs, nb_prims));
    sobjs.n1xs = d_n1xs;

    float *d_n1ys;
    CUDA_CONTROL(upload(d_n1ys, n1ys, nb_prims));
    sobjs.n1ys = d_n1ys;

    float *d_n1zs;
    CUDA_CONTROL(upload(d_n1zs, n1zs, nb_prims));
    sobjs.n1zs = d_n1zs;

    float *d_rads;
    CUDA_CONTROL(upload(d_rads, rads, nb_prims));
    sobjs.rads = d_rads;

    int *d_prim_g_indices;
    CUDA_CONTROL(upload(d_prim_g_indices,
                        prim_group_indices, nb_prims));
    sobjs.prim_group_indices = d_prim_g_indices;

    int *d_gstarts;
    CUDA_CONTROL(
        upload(d_gstarts, group_starts, nb_groups));
    sobjs.group_starts = d_gstarts;

    int *d_gsizes;
    CUDA_CONTROL(upload(d_gsizes, group_sizes, nb_groups));
    sobjs.group_sizes = d_gsizes;

    int *d_gids;
    CUDA_CONTROL(upload(d_gids, group_ids, nb_groups));
    sobjs.group_ids = d_gids;

    GroupType *d_gts;
    CUDA_CONTROL(upload(d_gts, gtypes, nb_groups));
    sobjs.gtypes = d_gts;
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
  }
  __host__ __device__ ScenePrim
  get_primitive(int gstart, int group_id) {
    HittableParam ht(
        htypes[gstart], p1xs[gstart], p1ys[gstart],
        p1zs[gstart], p2xs[gstart], p2ys[gstart],
        p2zs[gstart], n1xs[gstart], n1ys[gstart],
        n1zs[gstart], rads[gstart]);
    ImageParam imp(widths[gstart], heights[gstart],
                   bytes_per_pixels[gstart],
                   image_indices[gstart]);
    TextureParam tp(ttypes[gstart], cvals[gstart],
                    scales[gstart], imp);
    MaterialParam mp(tp, mtypes[gstart],
                     fuzz_ref_idxs[gstart]);
    ScenePrim prim(mp, ht, prim_group_indices[gstart],
                   group_id);
    return prim;
  }
};
