// libs
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
#include <kernelio.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>

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
  int WIDTH = 640;
  int HEIGHT = static_cast<int>(WIDTH / aspect_ratio);
  int BLOCK_WIDTH = 16;
  int BLOCK_HEIGHT = 8;
  int SAMPLE_NB = 300;
  int BOUNCE_NB = 150;

  bool gpu_io = false;

  get_device_props();

  std::cerr << "Resim boyutumuz " << WIDTH << "x" << HEIGHT
            << std::endl;

  std::cerr << BLOCK_WIDTH << "x" << BLOCK_HEIGHT
            << " bloklar halinde" << std::endl;

  // declare world
  SceneObjects sobjs = make_cornell_box();

  // declare camera
  Camera cam = makeCam(WIDTH, HEIGHT);

  //
  if (gpu_io) {
    gpu_main(WIDTH, HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT,
             SAMPLE_NB, BOUNCE_NB, aspect_ratio, sobjs,
             cam);
  } else {
    cpu_main(WIDTH, HEIGHT, SAMPLE_NB, BOUNCE_NB, cam,
             sobjs);
  }
}
