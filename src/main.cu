// libs
#include <camera.cuh>
#include <cbuffer.cuh>
#include <color.hpp>
#include <debug.hpp>
#include <external.hpp>
#include <hittables.cuh>
#include <makeworld.cuh>
#include <material.cuh>
#include <ray.cuh>
#include <sphere.cuh>
#include <texture.cuh>
#include <trace.cuh>
#include <vec3.cuh>

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

void get_device_props() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cerr << "Device Number: " << i << std::endl;
    std::cerr << "Device name: " << prop.name << std::endl;
    std::cerr << "Memory Clock Rate (KHz): "
              << prop.memoryClockRate << std::endl;
    std::cerr << "Memory Bus Width (bits): "
              << prop.memoryBusWidth << std::endl;
    std::cerr << "  Peak Memory Bandwidth (GB/s): "
              << 2.0 * prop.memoryClockRate *
                     (prop.memoryBusWidth / 8) / 1.0e6
              << std::endl;
  }
}

Camera makeCam(int WIDTH, int HEIGHT) {

  Vec3 lookfrom(278, 278, -800);
  Vec3 lookat(278, 278, 0);
  Vec3 wup(0, 1, 0);
  float vfov = 40.0f;
  float aspect_r = float(WIDTH) / float(HEIGHT);
  float dist_to_focus = (lookfrom - lookat).length();
  float aperture = 0.0;
  float t0 = 0.0f, t1 = 1.0f;

  Camera cam(lookfrom, lookat, wup, vfov, aspect_r,
             aperture, dist_to_focus, t0, t1);
  return cam;
}

void make_image(thrust::device_ptr<unsigned char> &imdata,
                thrust::device_ptr<int> &imwidths,
                thrust::device_ptr<int> &imhs,
                thrust::device_ptr<int> &imch) {
  std::vector<const char *> impaths = {"media/earthmap.png",
                                       "media/lsjimg.png"};
  std::vector<int> ws, hes, nbChannels;
  int totalSize;
  std::vector<unsigned char> imdata_h;
  imread(impaths, ws, hes, nbChannels, imdata_h, totalSize);
  ////// thrust::device_ptr<unsigned char> imda =
  //////    thrust::device_malloc<unsigned char>(imd.size);
  unsigned char *h_ptr = imdata_h.data();

  // --------------------- image ------------------------
  upload_thrust<unsigned char>(imdata, h_ptr,
                               (int)imdata_h.size());

  int *ws_ptr = ws.data();

  upload_thrust<int>(imwidths, ws_ptr, (int)ws.size());

  int *hs_ptr = hes.data();
  upload_thrust<int>(imhs, hs_ptr, (int)hes.size());

  int *nb_ptr = nbChannels.data();
  upload_thrust<int>(imch, nb_ptr, (int)nbChannels.size());
}

int main() {
  float aspect_ratio = 16.0f / 9.0f;
  int WIDTH = 320;
  int HEIGHT = static_cast<int>(WIDTH / aspect_ratio);
  int BLOCK_WIDTH = 32;
  int BLOCK_HEIGHT = 18;
  int SAMPLE_NB = 10;
  int BOUNCE_NB = 10;

  get_device_props();

  std::cerr << "Resim boyutumuz " << WIDTH << "x" << HEIGHT
            << std::endl;

  std::cerr << BLOCK_WIDTH << "x" << BLOCK_HEIGHT
            << " bloklar halinde" << std::endl;

  // declare frame size
  int total_pixel_size = WIDTH * HEIGHT;
  size_t frameSize = 3 * total_pixel_size;

  // declare frame
  thrust::device_ptr<Vec3> fb =
      thrust::device_malloc<Vec3>(frameSize);
  CUDA_CONTROL(cudaGetLastError());

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

  // declare world
  thrust::device_ptr<Hittables *> world;
  world = thrust::device_malloc<Hittables *>(1);
  SceneObjects sobjs = make_cornell_box();

  // make_final_world(hs, world);
  // make_cornell(hs, world, lshape);

  // declara imdata
  SceneObjects d_sobjs = sobjs.to_device();

  // --------------------- image ------------------------
  CUDA_CONTROL(cudaGetLastError());
  make_empty_c_box<<<1, 1>>>(d_sobjs);

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

  // declare camera
  Camera cam = makeCam(WIDTH, HEIGHT);
  //

  // render<<<blocks, threads>>>(
  //    thrust::raw_pointer_cast(fb), WIDTH, HEIGHT,
  //    SAMPLE_NB, BOUNCE_NB, cam,
  //    thrust::raw_pointer_cast(world),
  //    thrust::raw_pointer_cast(randState1));
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
  free_empty_cornell(fb, world, randState1, randState2);
  d_sobjs.d_free();
  CUDA_CONTROL(cudaGetLastError());

  cudaDeviceReset();
}
