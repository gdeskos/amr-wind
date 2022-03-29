#ifndef COMPLEX_TERRAIN_OPS_H
#define COMPLEX_TERRAIN_OPS_H

#include "amr-wind/immersed_boundary/complex_terrain/ComplexTerrain.H"
#include "amr-wind/immersed_boundary/IBOps.H"
#include "amr-wind/core/MultiParser.H"
#include "amr-wind/fvm/gradient.H"
#include "AMReX_Vector.H"
#include "amr-wind/wind_energy/actuator/actuator_utils.H"
#include "amr-wind/utilities/tensor_ops.H"

#include "AMReX_MultiFabUtil.H"

namespace amr_wind {
namespace ib {
namespace complex_terrain {

/** Parse common data for modeling non-blade bluff bodies via an immersed
 * bounary
 */
void read_inputs(
    ComplexTerrainBaseData&, IBInfo&, const ::amr_wind::utils::MultiParser&);

/** Initialize core data structures when modeling the IB geometry
 */
void init_data_structures(CFDSim&);

/** Set the velocity inside the IB Complex Terrain
 */
void apply_dirichlet_vel(CFDSim&, const amrex::Vector<amrex::Real>& vel_bc);

void compute_wall_stresses(
    CFDSim& sim, const amrex::Vector<amrex::Real>& tau_wall);

void prepare_netcdf_file(
    const std::string&, const ComplexTerrainBaseData&, const IBInfo&);

void write_netcdf(
    const std::string&,
    const ComplexTerrainBaseData&,
    const IBInfo&,
    const amrex::Real);

} // namespace complex_terrain

namespace ops {

template <typename GeomTrait>
struct UseDefaultOp
{
    static constexpr bool update_pos = true;
    static constexpr bool update_vel = true;
    static constexpr bool compute_force = true;
    static constexpr bool process_outputs = true;
};

template <typename GeomTrait>
struct UpdatePosOp<
    GeomTrait,
    typename std::enable_if<
        std::is_base_of<ComplexTerrainType, GeomTrait>::value>::type>
{
    void operator()(typename GeomTrait::DataType&) {}
};

template <typename GeomTrait>
struct UpdateVelOp<
    GeomTrait,
    typename std::enable_if<
        std::is_base_of<ComplexTerrainType, GeomTrait>::value>::type>
{
    void operator()(typename GeomTrait::DataType& data)
    {
        BL_PROFILE("amr-wind::ib::IB::update_velocities");

        const auto& wdata = data.meta();
        auto& sim = data.sim();

        complex_terrain::apply_dirichlet_vel(sim, wdata.vel_bc);
    }
};

template <typename GeomTrait>
struct ComputeForceOp<
    GeomTrait,
    typename std::enable_if<
        std::is_base_of<ComplexTerrainType, GeomTrait>::value>::type>
{
    void operator()(typename GeomTrait::DataType&) {}
};

template <typename IbType>
struct ComputeSourceOp<
    IbType,
    std::enable_if_t<std::is_base_of<ComplexTerrainType, IbType>::value>>
{
    void operator()(
        typename IbType::DataType& data,
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Geometry& geom)
    {
        Field& source = data.sim().repo().get_field("ib_src_term");
        const Field& level = data.sim().repo().get_field("ib_levelset");
        const Field& normal = data.sim().repo().get_field("ib_normal");
        const Field& velocity = data.sim().repo().get_field("velocity");
        const Field& density = data.sim().repo().get_field("density");

        const auto& bx = mfi.tilebox();
        const auto& varr = velocity(lev).array(mfi);
        const auto& darr = density(lev).array(mfi);
        const auto& phi_arr = level(lev).array(mfi);
        const auto& norm_arr = normal(lev).array(mfi);
        auto sarr = source(lev).array(mfi);

        const amrex::Real cd = data.meta().drag_coefficient;
        const amrex::Real eps = data.meta().eps_spread;

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Vector<amrex::Real> vel = {
                    {varr(i, j, k, 0), varr(i, j, k, 1), varr(i, j, k, 2)}};
                const amrex::Vector<amrex::Real> norm = {
                    {norm_arr(i, j, k, 0), norm_arr(i, j, k, 1),
                     norm_arr(i, j, k, 2)}};
                // Outside the band if the normal vector is assigned to be zero
                // normal vector will only be ~1 or ~0 so 0.5 is a safe value
                // that to confirm we are in the band where the wall model is
                // active
                const amrex::Real nmag = utils::vec_mag(norm.data());
                if (nmag > 0.5) {
                    const amrex::Real vdotn =
                        utils::dot_prod(norm.data(), vel.data());
                    const auto gauss_fac = actuator::utils::gaussian1d(
                        amrex::Math::abs(phi_arr(i, j, k)), eps);
                    // v_tan = v - (v dot n) n
                    amrex::Vector<amrex::Real> vtan = {
                        {vel[0] - vdotn * norm[0], vel[1] - vdotn * norm[1],
                         vel[2] - vdotn * norm[2]}};
                    // Get tangent velocity magnitude
                    const amrex::Real vtmag = utils::vec_mag(vtan.data());
                    // Calculate drag force C_D * 0.5 * rho * vtmag^2
                    // vtan already has one factor of vtmag in it, so there's no
                    // need to normalize, just leave off a factor in the
                    // coefficient.
                    const amrex::Real force =
                        -0.5 * darr(i, j, k) * gauss_fac * vtmag * cd;
                    sarr(i, j, k, 0) += force * vtan[0];
                    sarr(i, j, k, 1) += force * vtan[1];
                    sarr(i, j, k, 2) += force * vtan[2];
                }
            });
    }
};

template <typename GeomTrait>
struct ProcessOutputsOp<
    GeomTrait,
    typename std::enable_if<
        std::is_base_of<ComplexTerrainType, GeomTrait>::value>::type>
{
private:
    typename GeomTrait::DataType& m_data;

    std::string m_out_dir{""};
    std::string m_nc_filename{""};
    int m_out_freq{10};

public:
    explicit ProcessOutputsOp(typename GeomTrait::DataType& data) : m_data(data)
    {}

    void read_io_options(const ::amr_wind::utils::MultiParser& pp)
    {
        pp.query("output_frequency", m_out_freq);
    }

    void prepare_outputs(const std::string& out_dir)
    {
        m_nc_filename = out_dir + "/" + m_data.info().label + ".nc";
        complex_terrain::prepare_netcdf_file(
            m_nc_filename, m_data.meta(), m_data.info());
    }

    void write_outputs()
    {
        const auto& time = m_data.sim().time();
        const int tidx = time.time_index();
        if (tidx % m_out_freq != 0) return;

        complex_terrain::write_netcdf(
            m_nc_filename, m_data.meta(), m_data.info(), time.new_time());
    }
};

} // namespace ops

} // namespace ib
} // namespace amr_wind

#endif /* COMPLEX_TERRAIN_OPS_H */
