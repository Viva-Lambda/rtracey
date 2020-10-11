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
  TRIANGLE = 1,
  SPHERE = 2,
  MOVING_SPHERE = 3,
  XY_RECT = 4,
  XZ_RECT = 5,
  YZ_RECT = 6
};

enum GroupType : int { SOLID = 1, CONSTANT_MEDIUM = 2 };
