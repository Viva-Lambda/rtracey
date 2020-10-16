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
