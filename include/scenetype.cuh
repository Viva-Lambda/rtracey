#pragma once
#include <external.hpp>

enum PdfType : int {
  NONE_PDF = 0,
  COSINE_PDF = 1,
  HITTABLE_PDF = 2,
  MIXTURE_PDF = 3,
  PDF = 4
};
enum TextureType : int {
  NONE_TEXTURE = 0,
  SOLID_COLOR = 1,
  CHECKER = 2,
  NOISE = 3,
  IMAGE = 4,
  TEXTURE = 5
};
enum MaterialType : int {
  NONE_MATERIAL = 0,
  LAMBERTIAN = 1,
  METAL = 2,
  DIELECTRIC = 3,
  DIFFUSE_LIGHT = 4,
  ISOTROPIC = 5,
  MATERIAL = 6
};
enum HittableType : int {
  NONE_HITTABLE = 0,
  SPHERE = 1,
  MOVING_SPHERE = 2,
  RECTANGLE = 3,
  XY_RECT = 4,
  XZ_RECT = 5,
  YZ_RECT = 6,
  TRIANGLE = 7,
  XY_TRIANGLE = 8,
  XZ_TRIANGLE = 9,
  YZ_TRIANGLE = 10,
  HITTABLE = 11
};
enum GroupType : int {
  NONE_GRP = 0,
  BOX = 1,
  CONSTANT_MEDIUM = 2,
  SIMPLE_MESH = 3,
  OBJECT = 4,
  SCENE = 5
};
enum TransformationType : int {
  NONE_TRANSFORMATION = 0,
  ROTATE_Y = 1,
  TRANSLATE = 2,
  TRANSLATE_ROTATE = 2,
  SCALE = 3,
  SCALE_TRANSLATE = 5,
  SCALE_ROTATE = 6,
  SCALE_TRANSLATE_ROTATE = 7,
  TRANSFORMATION = 8
};
