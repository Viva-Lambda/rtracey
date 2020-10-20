#pragma once
#include <camera.cuh>
#include <cbuffer.cuh>
#include <color.hpp>
#include <debug.hpp>
#include <external.hpp>
#include <makeworld.cuh>
#include <ray.cuh>
#include <trace.cuh>
#include <vec3.cuh>
// scene objects
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
__global__ void rand_init(curandState *randState,
                          int seed) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(seed, 0, 0, randState);
  }
}

__global__ void render_init(int mx, int my,
                            curandState *randState,
                            int seed) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    curand_init(seed, 0, 0, randState);
  }
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= mx) || (j >= my)) {
    return;
  }
  int pixel_index = j * mx + i;
  // same seed, different index
  curand_init(seed + pixel_index, pixel_index, 0,
              &randState[pixel_index]);
}
//

int gpu_main(int WIDTH, int HEIGHT, int BLOCK_WIDTH,
             int BLOCK_HEIGHT, int SAMPLE_NB, int BOUNCE_NB,
             float aspect_ratio, SceneObjects &sobjs,
             Camera cam) {
  int total_pixel_size = WIDTH * HEIGHT;
  size_t frameSize = 3 * total_pixel_size;

  // declare frame
  thrust::device_ptr<Vec3> fb =
      thrust::device_malloc<Vec3>(frameSize);
  CUDA_CONTROL(cudaGetLastError());
  //
  // declare random state
  int SEED = time(NULL);
  thrust::device_ptr<curandState> randState1 =
      thrust::device_malloc<curandState>(frameSize);
  CUDA_CONTROL(cudaGetLastError());

  // declare random state 2
  thrust::device_ptr<curandState> randState2 =
      thrust::device_malloc<curandState>(1);
  CUDA_CONTROL(cudaGetLastError());
  rand_init<<<1, 1>>>(thrust::raw_pointer_cast(randState2),
                      SEED);
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  // declara imdata
  SceneObjects world = sobjs.to_device();

  // --------------------- image ------------------------
  CUDA_CONTROL(cudaGetLastError());
  make_cornell_box_k<<<1, 1>>>(
      world, thrust::raw_pointer_cast(randState2));

  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  clock_t baslar, biter;
  baslar = clock();

  dim3 blocks(WIDTH / BLOCK_WIDTH + 1,
              HEIGHT / BLOCK_HEIGHT + 1);
  dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
  render_init<<<blocks, threads>>>(
      WIDTH, HEIGHT, thrust::raw_pointer_cast(randState1),
      SEED + 7);
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(
      thrust::raw_pointer_cast(fb), WIDTH, HEIGHT,
      SAMPLE_NB, BOUNCE_NB, cam, world,
      thrust::raw_pointer_cast(randState1));
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());
  biter = clock();
  double saniyeler =
      ((double)(biter - baslar)) / CLOCKS_PER_SEC;
  std::cerr << "Islem " << saniyeler << " saniye surdu"
            << std::endl;

  std::cout << "P3" << std::endl;
  std::cout << WIDTH << " " << HEIGHT << std::endl;
  std::cout << "255" << std::endl;

  for (int j = HEIGHT - 1; j >= 0; j--) {
    for (int i = 0; i < WIDTH; i++) {
      size_t pixel_index = j * WIDTH + i;
      thrust::device_reference<Vec3> pix_ref =
          fb[pixel_index];
      Vec3 pixel = pix_ref;
      int ir = int(255.99 * pixel.r());
      int ig = int(255.99 * pixel.g());
      int ib = int(255.99 * pixel.b());
      std::cout << ir << " " << ig << " " << ib
                << std::endl;
    }
  }
  CUDA_CONTROL(cudaDeviceSynchronize());
  CUDA_CONTROL(cudaGetLastError());
  free_empty_cornell(fb, randState1, randState2);
  world.d_free();
  CUDA_CONTROL(cudaGetLastError());

  cudaDeviceReset();
  return 0;
}

int cpu_main(int WIDTH, int HEIGHT, int sample_nb,
             int bounceNb, Camera dcam,
             const SceneObjects &world) {
  clock_t baslar, biter;
  baslar = clock();

  std::cout << "P3" << std::endl;
  std::cout << WIDTH << " " << HEIGHT << std::endl;
  std::cout << "255" << std::endl;
  h_render(WIDTH, HEIGHT, sample_nb, bounceNb, dcam, world);
  biter = clock();
  double saniyeler =
      ((double)(biter - baslar)) / CLOCKS_PER_SEC;
  std::cerr << "Islem " << saniyeler << " saniye surdu"
            << std::endl;
  return 0;
}
