#include "bilaterial_denoise.h"
#include "normals.h"
#include "differential_geometry.h"
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
namespace pmp
{

BilaterialDenoise::BilaterialDenoise(DenoiseType eType)
{
    m_eDenoiseType = eType;
}

BilaterialDenoise::~BilaterialDenoise() 
{

}

void BilaterialDenoise::Denoise(SurfaceMesh& mesh)
{
    m_Mesh = mesh;
    m_vecFaceNormal = GetFaceNormal(mesh);
    m_vecFaceArea = GetFaceArea(mesh);
    m_vecFaceCentroid = GetFaceCentroid(mesh);

    if (!m_Mesh.n_vertices())
    {
        return;
    }

    std::vector<Normal> vecResult;
    std::cout << "BilaterialDenoise Denoise Begin!" << std::endl;
    if (m_eDenoiseType == DenoiseType::Bilaterial_Local)
    {
        LocalScheme(vecResult);
    }
    else if (m_eDenoiseType == DenoiseType::Bilaterial_Global)
    {
        GlobalScheme(vecResult);
    }
    std::cout << "BilaterialDenoise Denoise End!" << std::endl;

    std::cout << "BilaterialDenoise UpdateVertexPosition Begin!" << std::endl;
    UpdateVertexPosition(mesh, vecResult, 10, true);
    std::cout << "BilaterialDenoise UpdateVertexPosition End!" << std::endl;
}

void BilaterialDenoise::GlobalScheme(std::vector<Normal>& vecFilteredNormal) 
{
    vecFilteredNormal.resize(m_Mesh.n_faces());
    double dSmoothness = 0.01;

    std::vector<Eigen::Triplet<double>> coeff_triple;
    coeff_triple.clear();
    std::vector<Eigen::Triplet<double>> weight_triple;
    weight_triple.clear();
    Eigen::SparseMatrix<double> weight_matrix((int)m_Mesh.n_faces(),
                                              (int)m_Mesh.n_faces());
    Eigen::SparseMatrix<double> normalize_matrix((int)m_Mesh.n_faces(),
                                                 (int)m_Mesh.n_faces());
    Eigen::SparseMatrix<double> identity_matrix((int)m_Mesh.n_faces(),
                                                (int)m_Mesh.n_faces());
    identity_matrix.setIdentity();

    double dSigmaC = CaculateSigmaC(m_Mesh, m_vecFaceCentroid, 1.0);
    double dSigmaS = 0.35;
    std::cout << "dSigmaC: " << dSigmaC << std::endl;
    for (auto f : m_Mesh.faces())
    {
        int iIndexI = f.idx();
        Normal NormalI = m_vecFaceNormal[iIndexI];
        Point ptCentroidI = m_vecFaceCentroid[iIndexI];

        // openmesh也是利用半边是否有临近面判断的
        std::vector<Face> vecFace;
        auto halfedgeOfEachFace = m_Mesh.halfedges(f);
        for (auto h : halfedgeOfEachFace)
        {
            auto oh = m_Mesh.opposite_halfedge(h);
            auto ff = m_Mesh.face(oh);
            if (ff.is_valid())
            {
                vecFace.push_back(ff);
            }
        }

        double dWeightSum = 0.0;
        for (int i = 0; i < (int)vecFace.size(); i++)
        {
            int iIndexJ = vecFace[i].idx();
            Normal NormalJ = m_vecFaceNormal[iIndexJ];
            Point ptCentroidJ = m_vecFaceCentroid[iIndexJ];
            Scalar dSpatialDistance = norm(ptCentroidI - ptCentroidJ);
            double dSpatialWeight =
                std::exp(-0.5 * dSpatialDistance * dSpatialDistance /
                         (dSigmaC * dSigmaC));
            double dRangeDistance = norm(NormalI - NormalJ);
            double dRangeWeight = std::exp(
                -0.5 * dRangeDistance * dRangeDistance / (dSigmaS * dSigmaS));
            double dWeight =
                m_vecFaceArea[iIndexJ] * dSpatialWeight * dRangeWeight;
            coeff_triple.push_back(
                Eigen::Triplet<double>(iIndexI, iIndexJ, dWeight));
            dWeightSum += dWeight;
        }
        if (dWeightSum)
        {
            weight_triple.push_back(
                Eigen::Triplet<double>(iIndexI, iIndexI, 1.0 / dWeightSum));
        }
    }
    weight_matrix.setFromTriplets(coeff_triple.begin(), coeff_triple.end());
    normalize_matrix.setFromTriplets(weight_triple.begin(),
                                     weight_triple.end());
    Eigen::SparseMatrix<double> matrix =
        identity_matrix - normalize_matrix * weight_matrix;
    Eigen::SparseMatrix<double> coeff_matrix =
        (1 - dSmoothness) * matrix.transpose() * matrix +
        dSmoothness * identity_matrix;

    Eigen::MatrixXd right_term(m_Mesh.n_faces(), 3);
    for (int i = 0; i < (int)m_vecFaceNormal.size(); i++)
    {
        right_term(i, 0) = m_vecFaceNormal[i][0];
        right_term(i, 1) = m_vecFaceNormal[i][1];
        right_term(i, 2) = m_vecFaceNormal[i][2];
    }
    right_term = dSmoothness * right_term;

    // solve Ax = b
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(coeff_matrix);
    solver.factorize(coeff_matrix);
    Eigen::MatrixX3d filtered_normals_matrix = solver.solve(right_term);
    filtered_normals_matrix.rowwise().normalize();
    for (int i = 0; i < (int)vecFilteredNormal.size(); i++)
    {
        vecFilteredNormal[i][0] = filtered_normals_matrix(i, 0);
        vecFilteredNormal[i][1] = filtered_normals_matrix(i, 1);
        vecFilteredNormal[i][2] = filtered_normals_matrix(i, 2);
    }
}

double BilaterialDenoise::CaculateSigmaC(SurfaceMesh& mesh, std::vector<Point> vecFaceCentroid, double dValue)
{
    double dSigmaC = 0.0;
    double dNum = 0.0;
    for (auto f : m_Mesh.faces())
    {
        Point ci = vecFaceCentroid[f.idx()];
        auto xxx = f.idx();
        auto halfedgeOfEachFace = m_Mesh.halfedges(f);
        for (auto h : halfedgeOfEachFace)
        {
            auto oh = m_Mesh.opposite_halfedge(h);
            auto ff = m_Mesh.face(oh);
            if (ff.is_valid())
            {
                Point cj = vecFaceCentroid[ff.idx()];
                dSigmaC += norm(ci - cj);
                dNum++;
            }
        }
    }
    dSigmaC *= dValue / dNum;
    return dSigmaC;
}

void BilaterialDenoise::UpdateVertexPosition(SurfaceMesh& mesh, std::vector<Normal>& vecFilteredNormal, int iIterationNumber, bool bFixedBoundary)
{
    auto vpoint = mesh.vertex_property<Point>("v:point");
    std::vector<Point> vecNewVertex(mesh.vertices_size());
    std::vector<Point> vecCentroid(mesh.faces_size(), Point(0.0, 0.0, 0.0));

    for (int i = 0; i < iIterationNumber; i++)
    {
        for (auto f : m_Mesh.faces())
        {
            vecCentroid[f.idx()] = centroid(m_Mesh, f);
        }

        for (auto v : m_Mesh.vertices())
        {
            auto p = vpoint[v];
            if (bFixedBoundary && mesh.is_boundary(v))
            {
                vecNewVertex[v.idx()] = p;
            }
            else
            {
                double dFaceNum = 0.0;
                Point temp(0.0, 0.0, 0.0);
                for (auto f : mesh.faces(v))
                {
                    auto tempNormal = vecFilteredNormal[f.idx()];
                    auto tempCentroid = vecCentroid[f.idx()];
                    temp += tempNormal * dot(tempNormal, tempCentroid - p);
                    dFaceNum++;
                }
                p += temp / dFaceNum;
                vecNewVertex[v.idx()] = p;
            }
        }
        for (auto v : m_Mesh.vertices())
        {
            vpoint[v] = vecNewVertex[v.idx()];
        }
    }
    mesh.garbage_collection();
}

void BilaterialDenoise::LocalScheme(std::vector<Normal>& vecFilteredNormal)
{
    vecFilteredNormal.resize(m_Mesh.n_faces());

    int iNormalIterationNumber = 20;
    double dSigmaC = CaculateSigmaC(m_Mesh, m_vecFaceCentroid, 1.0);
    double dSigmaS = 0.35;
    std::cout << "dSigmaC: " << dSigmaC << std::endl;

    for (int i = 0; i < iNormalIterationNumber; i++)
    {
        for (auto f : m_Mesh.faces())
        {
            int iIndexI = f.idx();
            Normal NormalI = m_vecFaceNormal[iIndexI];
            Point ptCentroidI = m_vecFaceCentroid[iIndexI];

            std::vector<Face> vecFace;
            auto halfedgeOfEachFace = m_Mesh.halfedges(f);
            for (auto h : halfedgeOfEachFace)
            {
                auto oh = m_Mesh.opposite_halfedge(h);
                auto ff = m_Mesh.face(oh);
                if (ff.is_valid())
                {
                    vecFace.push_back(ff);
                }
            }

            Normal tempNormal(0.0, 0.0, 0.0);
            double dWeightSum = 0.0;
            for (int i = 0; i < vecFace.size(); i++)
            {
                int iIndexJ = vecFace[i].idx();
                Normal NormalJ = m_vecFaceNormal[iIndexJ];
                Point ptCentroidJ = m_vecFaceCentroid[iIndexJ];

                double dSpatialDistance = norm(ptCentroidI - ptCentroidJ);
                double dSpatialWeight =
                    std::exp(-0.5 * dSpatialDistance * dSpatialDistance /
                             (dSigmaC * dSigmaC));
                double dRangeDistance = norm(NormalI - NormalJ);
                double dRangeWeight =
                    std::exp(-0.5 * dRangeDistance * dRangeDistance /
                             (dSigmaS * dSigmaS));

                double dWeight =
                    m_vecFaceArea[iIndexJ] * dSpatialWeight * dRangeWeight;
                dWeightSum += dWeight;
                tempNormal += NormalJ * dWeight;
            }
            tempNormal /= dWeightSum;
            tempNormal.normalize();
            vecFilteredNormal[iIndexI] = tempNormal;
        }
        m_vecFaceNormal = vecFilteredNormal;
    }
}

void BilaterialDenoiseExecute(SurfaceMesh& mesh)
{
    //BilaterialDenoise BilDenoiseInstance(DenoiseType::Bilaterial_Global);
    //std::vector<Normal> vecFilteredNormal;
    //std::cout << "BilaterialDenoise Denoise Begin!" << std::endl;
    //BilDenoiseInstance.GlobalScheme(vecFilteredNormal);
    //std::cout << "BilaterialDenoise Denoise End!" << std::endl;

    //std::cout << "BilaterialDenoise UpdateVertexPosition Begin!" << std::endl;
    //BilDenoiseInstance.UpdateVertexPosition(mesh, vecFilteredNormal, 10, true);
    //std::cout << "BilaterialDenoise UpdateVertexPosition End!" << std::endl;
}

}


