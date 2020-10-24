#pragma once
// mesh
#include <external.hpp>
#include <texparam.cuh>
#include <vec3.cuh>

struct Mesh {
  int psize;
  Point3 *p1s, *p2s, *p3s;
  unsigned int *indices;
  ImageParam *imparams;
  int imsize;
  __host__ __device__ Mesh()
      : p1s(nullptr), p2s(nullptr), p3s(nullptr),
        indices(nullptr), imparams(nullptr), psize(0),
        imsize(0) {}

  __host__ __device__ Mesh(Point3 *_p1s, Point3 *_p2s,
                           Point3 *_p3s, int _psize,
                           unsigned int *idxs,
                           ImageParam *imps, int ims)
      : psize(_psize), imsize(ims) {
    deepcopy(p1s, _p1s, psize);
    deepcopy(p2s, _p2s, psize);
    deepcopy(p3s, _p3s, psize);
    deepcopy(imparams, imps, imsize);
  }
};

struct Model {
  std::vector<Mesh> meshes;
  int msize;
  __host__ void loadModel(std::string mpath);
  __host__ void processNode(aiNode *node,
                            const aiScene *scene);
  __host__ Mesh processMesh(aiMesh *mesh,
                            const aiScene *scene);
  __host__ std::vector<ImageParam>
  loadMaterialTextures(aiMaterial *mat, aiTextureType type,
                       std::string typeName);
  __host__ Model(std::string mpath) {
    loadModel(mpath);
    msize = static_cast<int>(meshes.size());
  }
};
void Model::loadModel(std::string mpath) {

  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(
      mpath,
      aiProcess_Triangulate | aiProcess_CalcTangentSpace);
  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    std::cout << "ERROR::ASSIMP::"
              << importer.GetErrorString() << std::endl;
    return;
  }
  directory = path.substr(0, path.find_last_of('/'));

  // start processing from root node recursively
  processNode(scene->mRootNode, scene);
}
void Model::processNode(aiNode *node,
                        const aiScene *scene) {
  // process the meshes of the given node on scene
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    meshes.push_back(this->processMesh(mesh, scene));
  }
  // now all meshes of this node has been processed
  // we should continue to meshes of child nodes
  for (unsigned int k = 0; k < node->mNumChildren; k++) {
    processNode(node->mChildren[k], scene);
  }
}
Mesh Model::processMesh(aiMesh *mesh,
                        const aiScene *scene) {
  // data
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  std::vector<Texture> textures;

  // iteratin on vertices of the mesh
  std::vector<std::vector<Point3>> triangles;
  for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
    aiFace triangle_face = mesh->mFaces[j];
    std::vector<Point3> triangle;
    for (unsigned int i = 0; i < triangle_face.mNumIndices;
         i++) {
      unsigned int index = triangle_face.mIndices[i];
      Point3 vec;
      vec[0] = mesh->mVertices[index].x;
      vec[1] = mesh->mVertices[index].y;
      vec[2] = mesh->mVertices[index].z;
      triangle.push_back(vec);
    }
    triangles.push_back(triangle);
  }
  // vertice iteration done now we should deal with indices
  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    for (unsigned int k = 0; k < face.mNumIndices; k++) {
      indices.push_back(face.mIndices[k]);
    }
    // now deal with materials
    if (mesh->mMaterialIndex >= 0) {
      aiMaterial *material =
          scene->mMaterials[mesh->mMaterialIndex];
      // we retrieve textures
      // 1. diffuse maps
      std::vector<Texture> diffuseMaps =
          this->loadMaterialTextures(material,
                                     aiTextureType_DIFFUSE,
                                     "texture_diffuse");
      textures.insert(textures.end(), diffuseMaps.begin(),
                      diffuseMaps.end());
      // 2. specular maps
      std::vector<Texture> specularMaps =
          this->loadMaterialTextures(material,
                                     aiTextureType_SPECULAR,
                                     "texture_specular");
      textures.insert(textures.end(), specularMaps.begin(),
                      specularMaps.end());
      // 3. normal maps
      std::vector<Texture> normalMaps =
          this->loadMaterialTextures(material,
                                     aiTextureType_HEIGHT,
                                     "texture_normal");
      textures.insert(textures.end(), normalMaps.begin(),
                      normalMaps.end());

      // 4. height maps
      std::vector<Texture> heightMaps =
          this->loadMaterialTextures(material,
                                     aiTextureType_AMBIENT,
                                     "texture_height");
      textures.insert(textures.end(), heightMaps.begin(),
                      heightMaps.end());
    }
  }
  return Mesh(vertices, indices, textures);
}
