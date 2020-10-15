#pragma once

enum TextureType : int {
  NONE_TEXTURE = 0,
  SOLID_COLOR = 1,
  CHECKER = 2,
  NOISE = 3,
  IMAGE = 4
};

enum MaterialType : int {
  NONE_MATERIAL = 0,
  LAMBERTIAN = 1,
  METAL = 2,
  DIELECTRIC = 3,
  DIFFUSE_LIGHT = 4,
  ISOTROPIC = 5
};

enum HittableType : int {
  NONE_HITTABLE = 0,
  SPHERE = 1,
  MOVING_SPHERE = 2,
  XY_RECT = 3,
  XZ_RECT = 4,
  YZ_RECT = 5,
  XY_TRIANGLE = 6,
  XZ_TRIANGLE = 7,
  YZ_TRIANGLE = 8
};

enum GroupType : int {
  BOX = 1,
  CONSTANT_MEDIUM = 2,
  SIMPLE_MESH = 3
};
