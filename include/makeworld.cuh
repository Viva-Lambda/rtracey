// make world kernel
#pragma once

#include <ray.cuh>
#include <scenegroup.cuh>
#include <sceneobj.cuh>
#include <sceneparam.cuh>
#include <sceneprim.cuh>
#include <vec3.cuh>

SceneObjects make_cornell_box() {
  TextureParam red_solid =
      mkSolidColorParam(Color(.65, .05, .05));
  TextureParam green_solid =
      mkSolidColorParam(Color(.12, .45, .15));
  TextureParam blue_solid =
      mkSolidColorParam(Color(.05, .05, .65));
  TextureParam white_solid =
      mkSolidColorParam(Color(.75, .75, .75));
  TextureParam light_solid =
      mkSolidColorParam(Color(15, 15, 15));

  MaterialParam red_param = mkLambertParam(red_solid);
  MaterialParam green_param =
      mkMetalParam(green_solid, 0.3f);
  MaterialParam blue_param = mkLambertParam(blue_solid);
  MaterialParam white_param = mkLambertParam(white_solid);
  MaterialParam light_param =
      mkDiffuseLightParam(light_solid);

  HittableParam green_wall =
      mkYZRectHittable(0, 555, 0, 555, 555);
  auto red_wall = mkYZRectHittable(0, 555, 0, 555, 0);
  auto light_wall =
      mkXZRectHittable(213, 343, 227, 332, 554);
  auto white_wall = mkXZRectHittable(0, 555, 0, 555, 0);
  auto white_wall2 = mkXZRectHittable(0, 555, 0, 555, 555);
  auto blue_wall = mkXYRectHittable(0, 555, 0, 555, 555);
  //
  int prim_count = 6;
  ScenePrim *prms = new ScenePrim[prim_count];
  //
  int pcount = 0;
  prms[pcount] = ScenePrim(green_param, green_wall, 0, 0);
  pcount++;
  prms[pcount] = ScenePrim(red_param, red_wall, 1, 0);
  pcount++;
  prms[pcount] = ScenePrim(light_param, light_wall, 2, 0);
  pcount++;
  prms[pcount] = ScenePrim(white_param, white_wall, 3, 0);
  pcount++;
  prms[pcount] = ScenePrim(white_param, white_wall2, 4, 0);
  pcount++;
  prms[pcount] = ScenePrim(blue_param, blue_wall, 5, 0);
  TextureParam tp;
  SceneGroup sg(prms, prim_count, 0, SOLID, 0.0f, tp);
  SceneGroup sgs[] = {sg};
  SceneObjects sobjs(sgs, 1);
  Hittables hs = sobjs.to_hittables();
  return sobjs;
}

__global__ void make_empty_c_box(SceneObjects sobjs) {
  //
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    SceneGroup sg;
    sobjs.fill_group(sg, 0);
    // Hittable **hg = sg.to_hittable_list();
    ScenePrim sp = sobjs.get_primitive(1, 0);
    Hittable *hi = sp.to_hittable();
  }
}

//__global__ void make_empty_cornell_box(Hittables **world,
//                                     Hittable **ss) {
// declare objects
// if (threadIdx.x == 0 && blockIdx.x == 0) {

//  Material *red = new Lambertian(Color(.65, .05, .05));
//  Material *blue = new Lambertian(Color(.05, .05, .65));
//  Material *white = new Lambertian(Color(.73, .73, .73));
//  Material *green = new Metal(Color(.12, .45, .15), 0.3);
//  Material *light = new DiffuseLight(Color(15, 15, 15));

//  // ----------- Groups --------------------
//  Hittable **groups = new Hittable *[3];

//  int obj_count = 0;
//  int group_count = 0;
//  // --------------- cornell box group ----------------

//  ss[obj_count] = new YZRect(0, 555, 0, 555, 555, green);
//  obj_count++;
//  ss[obj_count] = new YZRect(0, 555, 0, 555, 0, red);
//  obj_count++;
//  ss[obj_count] =
//      new XZRect(213, 343, 227, 332, 554, light);
//  obj_count++;
//  ss[obj_count] = new XZRect(0, 555, 0, 555, 0, white);
//  obj_count++;
//  ss[obj_count] = new XZRect(0, 555, 0, 555, 555, white);
//  obj_count++;
//  ss[obj_count] = new XYRect(0, 555, 0, 555, 555, blue);

//  Hittable *c_box =
//      new HittableGroup(ss, 0, obj_count + 1);
//  groups[group_count] = c_box;

//  // -------------- Boxes -------------------------

//  obj_count++;
//  Point3 bp1(0.0f);
//  Point3 bp2(165, 330, 165);
//  Box b1(bp1, bp2, white, ss, obj_count);
//  b1.rotate_y(ss, 15.0f);
//  b1.translate(ss, Vec3(265, 0, 295));
//  // b1.to_gas(0.01f, &randState, Color(1.0, 0.3, 0.7),
//  // ss);
//  Hittable *tall_box =
//      new HittableGroup(ss, b1.start_index, b1.end_index);

//  Hittable *smoke_box1 = new ConstantMedium(
//      tall_box, 0.01, Color(0.8f, 0.2, 0.4), randState);

//  group_count++;

//  groups[group_count] = smoke_box1;

//  obj_count++;
//  Point3 bp3(0.0f);
//  Point3 bp4(165.0f);
//  Box b2(bp3, bp4, white, ss, obj_count);
//  b2.rotate_y(ss, -18.0f);
//  b2.translate(ss, Point3(130, 0, 165));
//  obj_count++;

//  Hittable *short_box =
//      new HittableGroup(ss, b2.start_index, b2.end_index);
//  Hittable *smoke_box2 = new ConstantMedium(
//      short_box, 0.01, Color(0.8f, 0.3, 0.8), randState);

//  group_count++;
//  groups[group_count] = smoke_box2;

//  group_count++;
//  order_scene(groups, group_count);

//  world[0] = new Hittables(groups, group_count);
//}
//}

void free_empty_cornell(
    thrust::device_ptr<Vec3> &fb,
    thrust::device_ptr<Hittables *> &world,
    thrust::device_ptr<curandState> randState1,
    thrust::device_ptr<curandState> randState2) {

  thrust::device_free(fb);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(world);
  CUDA_CONTROL(cudaGetLastError());

  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState2);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState1);
  CUDA_CONTROL(cudaGetLastError());
}
