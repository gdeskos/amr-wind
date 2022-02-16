#include "aw_test_utils/AmrexTest.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Gpu.H"
#include "../../amr-wind/immersed_boundary/bluff_body/ghost_cell.H"

namespace amr_wind_tests {

namespace {

void print_vandermonde(amrex::Array2D<amrex::Real, 0, 7, 0, 7>& Vmatrix)
{
    int Np = 8;

    amrex::Print() << "\n Vandermonde Matrix:";

    for (int row = 0; row < Np; row++) {
        amrex::Print() << "[";
        for (int col = 0; col < Np; col++) {
            amrex::Print() << Vmatrix(row, col) << " ";
        }
        amrex::Print() << "]\n";
    }
}

void print_coeffvector(amrex::Vector<amrex::Real>& coeff_vector)
{
    int Np = 8;

    amrex::Print() << "\n Coeff Vector:\n [";

    for (int i = 0; i < Np; i++) {
        amrex::Print() << coeff_vector[i] << " ";
    }

    amrex::Print() << "]\n\n";
}

amrex::Real L2_Norm(amrex::Vector<amrex::Real>& coeff_vector)
{
    int Np = 8;

    amrex::Real norm = 0.0;

    for (int i = 0; i < Np - 1; i++) {
        norm = norm + std::pow(coeff_vector[i], 2);
    }

    norm = std::sqrt(norm);

    return norm;
}

} // namespace

class GaussSiedelMethodTest : public AmrexTest
{};

TEST_F(GaussSiedelMethodTest, vandermonde)
{
    constexpr double tol = 1.0e-3;

    const amrex::Real exact_norm = 1.0840; // From Matlab

    amrex::Real coeff_norm;

    amrex::Array2D<amrex::Real, 0, 7, 0, 7> Vmatrix;

    const amrex::Vector<amrex::Real> x_loc_data{0.5,  0.5,  0.5,  0.5,
                                                -0.5, -0.5, -0.5, -0.5};
    const amrex::Vector<amrex::Real> y_loc_data{0.5, -0.5, -0.5, 0.5,
                                                0.5, -0.5, -0.5, 0.5};
    const amrex::Vector<amrex::Real> z_loc_data{0.5, 0.5, -0.5, -0.5,
                                                0.5, 0.5, -0.5, -0.5};
    const amrex::Vector<amrex::Real> phi{0.5, 0.1, 0.3, 0.7,
                                         0.5, 1.0, 0.8, 0.5};

    // Initial guess for coeff_vector
    amrex::Vector<amrex::Real> coeff_vector{0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0};

    // Form the Vandermonde matrix
    amr_wind::ib::vandermonde_matrix(
        x_loc_data, y_loc_data, z_loc_data, Vmatrix);

    print_vandermonde(Vmatrix);

    // Gauss-Siedel method to compute coefficient vector
    amr_wind::ib::gauss_siedel_coeffVector(Vmatrix, phi, coeff_vector);

    print_coeffvector(coeff_vector);

    // L2 Norm of coefficient vector
    coeff_norm = L2_Norm(coeff_vector);

    // Verify if the matrix inversion is correct
    EXPECT_NEAR(coeff_norm, exact_norm, tol);
}

} // namespace amr_wind_tests
