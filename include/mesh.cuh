#pragma once
// mesh
#include <external.hpp>
#include <texparam.cuh>
#include <utils.cuh>
#include <vec3.cuh>

struct MeshTriangle {
  Point3 p1, p2, p3;
  __host__ __device__ MeshTriangle() {}
  __host__ MeshTriangle(std::vector<Point3> ps) {
    p1 = ps[0];
    p2 = ps[1];
    p3 = ps[2];
  }
  __host__ __device__ Primitive
  to_primitive(int indx, int mesh_id, MaterialParam mpar) {
    HittableParam tri_hparam = mkTriangle(p1, p2, p3);
    Primitive prim(mpar, tri_hparam, indx, mesh_id);
    return prim;
  }
};

struct Mesh {
  int tricount;
  MeshTriangle *tris;
  int mesh_id;
  __host__ __device__ Mesh()
      : tris(nullptr), tricount(0), mesh_id(0) {}

  __host__ __device__ Mesh(MeshTriangle *_tris, int _psize,
                           int mid)
      : tricount(_psize), mesh_id(mid) {
    deepcopy(tris, _tris, _psize);
  }
  __host__ __device__ void m_free() { delete[] tris; }
  __host__ __device__ GroupParam to_group(float d,
                                          MaterialParam m) {
    Primitive *prims = new Primitive[tricount];
    for (int i = 0; i < tricount; i++) {
      MeshTriangle mtri = tris[i];
      prims[i] = mtri.to_primitive(i, mesh_id, m);
    }
    GroupParam gp(prims, tricount, mesh_id, SIMPLE_MESH, d,
                  m);
    return gp;
  }
};
struct Model {
  Mesh *meshes;
  int mcount;
  __host__ void loadModel(std::string mpath) {

    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(
        mpath,
        aiProcess_Triangulate | aiProcess_CalcTangentSpace);
    if (!scene ||
        scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
        !scene->mRootNode) {
      std::cout << "ERROR::ASSIMP::"
                << importer.GetErrorString() << std::endl;
      return;
    }
    // start processing from root node recursively
    std::vector<Mesh> mesh;
    processNode(scene->mRootNode, scene, 0, mesh);
    meshes = mesh.data();
    deepcopy(meshes, mesh.data(), (int)mesh.size());
    mcount = (int)mesh.size();
  }

  __host__ void processNode(aiNode *node,
                            const aiScene *scene,
                            int node_id,
                            std::vector<Mesh> &msh) {
    // process the meshes of the given node on scene
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
      aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
      msh.push_back(processMesh(mesh, scene, i + node_id));
    }
    // now all meshes of this node has been processed
    // we should continue to meshes of child nodes
    for (unsigned int k = 0; k < node->mNumChildren; k++) {
      processNode(node->mChildren[k], scene, k + node_id,
                  msh);
    }
  }
  __host__ Mesh processMesh(aiMesh *mesh,
                            const aiScene *scene,
                            int mesh_id) {
    // iteratin on vertices of the mesh
    std::vector<MeshTriangle> triangles;
    for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
      aiFace triangle_face = mesh->mFaces[j];
      std::vector<Point3> triangle;
      for (unsigned int i = 0;
           i < triangle_face.mNumIndices; i++) {
        unsigned int index = triangle_face.mIndices[i];
        Point3 vec;
        vec[0] = mesh->mVertices[index].x;
        vec[1] = mesh->mVertices[index].y;
        vec[2] = mesh->mVertices[index].z;
        triangle.push_back(vec);
      }
      MeshTriangle tri(triangle);
      triangles.push_back(tri);
    }
    Mesh m(triangles.data(), (int)triangles.size(),
           mesh_id);
    return m;
  }
  __host__ Model(std::string mpath) { loadModel(mpath); }
  __host__ __device__ void to_groups(MaterialParam mp,
                                     GroupParam *gp) {
    for (int i = 0; i < mcount; i++) {
      Mesh m = meshes[i];
      GroupParam g = m.to_group(0.0f, mp);
      gp[i] = g;
    }
  }
};
