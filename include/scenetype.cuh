#pragma once

enum TextureType : int {
  SOLID_COLOR = 1,
  CHECKER = 2,
  NOISE = 3,
  IMAGE = 4
};

enum MaterialType : int {
  LAMBERTIAN = 1,
  METAL = 2,
  DIELECTRIC = 3,
  DIFFUSE_LIGHT = 4,
  ISOTROPIC = 5
};

enum HittableType : int {
  SPHERE = 1,
  MOVING_SPHERE = 2,
  XY_RECT = 3,
  XZ_RECT = 4,
  YZ_RECT = 5,
  XY_TRIANGLE = 6,
  XZ_TRIANGLE = 7,
  YZ_TRIANGLE = 8
};

enum GroupType : int { SOLID = 1, CONSTANT_MEDIUM = 2 };
