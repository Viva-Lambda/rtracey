#pragma once

#include <assert.h>

#include <assimp/Importer.hpp>
#include <assimp/material.h>
//#includep/pbrmaterial.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/types.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <sstream>
#include <thrust/copy.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/swap.h>
#include <time.h>

#include <math.h>
#include <random>
#include <stdlib.h>

// stb image read & write
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
