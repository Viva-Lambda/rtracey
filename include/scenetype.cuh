#pragma once
#include <external.hpp>

__managed__ int NONE_TEXTURE = 0;
__managed__ int SOLID_COLOR = 1;
__managed__ int CHECKER = 2;
__managed__ int NOISE = 3;
__managed__ int IMAGE = 4;

__managed__ int NONE_MATERIAL = 0;
__managed__ int LAMBERTIAN = 1;
__managed__ int METAL = 2;
__managed__ int DIELECTRIC = 3;
__managed__ int DIFFUSE_LIGHT = 4;
__managed__ int ISOTROPIC = 5;

__managed__ int NONE_HITTABLE = 0;
__managed__ int SPHERE = 1;
__managed__ int MOVING_SPHERE = 2;
__managed__ int XY_RECT = 3;
__managed__ int XZ_RECT = 4;
__managed__ int YZ_RECT = 5;
__managed__ int XY_TRIANGLE = 6;
__managed__ int XZ_TRIANGLE = 7;
__managed__ int YZ_TRIANGLE = 8;

__managed__ int BOX = 1;
__managed__ int CONSTANT_MEDIUM = 2;
__managed__ int SIMPLE_MESH = 3;

enum TextureType : int {
  NONE_TET = 0,
  SOLID_TET = 1,
  CHECKER_TET = 2,
  NOISE_TET = 3,
  IMAGE_TET = 4,
  TEXTURE = 5
};
enum MaterialType : int {
  NONE_MAT = 0,
  LAMBERTIAN_MAT = 1,
  METAL_MAT = 2,
  DIELECTRIC_MAT = 3,
  DIFFUSE_LIGHT_MAT = 4,
  ISOTROPIC_MAT = 5,
  MATERIAL = 6
};
enum HittableType : int {
  NONE_HIT = 0,
  SPHERE_HIT = 1,
  MOVING_SPHERE_HIT = 2,
  RECT_HIT = 3,
  XY_RECT_HIT = 4,
  XZ_RECT_HIT = 5,
  YZ_RECT_HIT = 6,
  TRIANGLE_HIT = 7,
  XY_TRIANGLE_HIT = 8,
  XZ_TRIANGLE_HIT = 9,
  YZ_TRIANGLE_HIT = 10,
  HITTABLE = 11
};
enum GroupType : int {
  NONE_GRP = 0,
  BOX_GRP = 1,
  CONSTANT_MEDIUM_GRP = 2,
  SIMPLE_MESH_GRP = 3,
  SCENE_GRP = 4
};
