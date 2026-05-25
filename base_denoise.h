#pragma once
#include "pmp/surface_mesh.h"
#include "normals.h"
#include "differential_geometry.h"

namespace pmp 
{

enum class DenoiseType
{
    Bilaterial_Local,
    Bilaterial_Global,
    Unknown
};

class BaseDenoise
{
public:
    BaseDenoise(){};
    ~BaseDenoise(){};

public:
    virtual void Denoise(SurfaceMesh& mesh) = 0;

public:
    std::vector<Normal> GetFaceNormal(const SurfaceMesh& mesh)
    {
        std::vector<Normal> vecFaceNormal(mesh.faces_size());
        for (auto f : mesh.faces())
        {
            vecFaceNormal[f.idx()] = face_normal(mesh, f);
        }
        return vecFaceNormal;
    }

    std::vector<Scalar> GetFaceArea(const SurfaceMesh& mesh)
    {
        std::vector<Scalar> vecFaceArea(mesh.faces_size());
        for (auto f : mesh.faces())
        {
            vecFaceArea[f.idx()] = face_area(mesh, f);
        }
        return vecFaceArea;
    }

    std::vector<Point> GetFaceCentroid(const SurfaceMesh& mesh)
    {
        std::vector<Point> vecFaceCentroid(mesh.faces_size());
        for (auto f : mesh.faces())
        {
            vecFaceCentroid[f.idx()] = centroid(mesh, f);
        }
        return vecFaceCentroid;
    }
};

}

