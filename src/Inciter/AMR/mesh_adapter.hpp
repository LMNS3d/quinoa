#ifndef QUINOA_MESH_ADAPTER_H
#define QUINOA_MESH_ADAPTER_H

#include <stddef.h>
#include <vector>

#include "DerivedData.hpp"

#include "AMR_types.hpp"
#include "tet_store.hpp"
#include "node_connectivity.hpp"

#ifdef ENABLE_NODE_STORE
#include "node_store.hpp"
#endif

#include "refinement.hpp"
//#include "derefinement.hpp"

#include "Refinement_State.hpp"

namespace AMR {
    class mesh_adapter_t {

        public:

            //! Default constructor for migration
            mesh_adapter_t() {}

            //! Constructor taking a mesh graph
            explicit mesh_adapter_t( const std::vector< std::size_t >& inpoel ) :
                node_connectivity( tk::npoin_in_graph(inpoel) )
            {
                consume_tets( inpoel );
                tet_store.generate_edges();
            }

            void init_node_store(coord_type* m_x, coord_type* m_y, coord_type* m_z);

            // FIXME: Set these in a better way
            real_t derefinement_cut_off = 0.2;
            real_t refinement_cut_off = 0.9;

            AMR::tet_store_t tet_store;
            AMR::node_connectivity_t node_connectivity;

#ifdef ENABLE_NODE_STORE
            // for coord tracking type stuff (debugging)
            AMR::node_store_t node_store;
#endif

            AMR::refinement_t refiner;

            void consume_tets(const std::vector<std::size_t>& tetinpoel );

            void evaluate_error_estimate();
            void mark_uniform_refinement();
            void mark_uniform_derefinement();
            void mark_error_refinement(
              const std::vector< std::pair< edge_t, edge_tag > >& remote );

            void mark_error_refinement_corr( const EdgeData& edges );
            int detect_compatibility(
                    int num_locked_edges,
                    int num_intermediate_edges,
                    AMR::Refinement_Case refinement_case,
                    int normal=0
            );

            void lock_intermediates();

            void mark_refinement();
            void perform_refinement();

            void refinement_class_one(int num_to_refine, size_t tet_id);
            void refinement_class_two(edge_list_t edge_list, size_t tet_id);
            void refinement_class_three(size_t tet_id);

            void lock_tet_edges(size_t tet_id);
            void deactivate_tet_edges(size_t tet_id);
            bool check_valid_refinement_case(size_t child_id);

            void mark_derefinement();
            void perform_derefinement();
            //std::vector< std::size_t >& get_active_inpoel();

            void print_tets();

            void reset_intermediate_edges();
            void update_tet_edges_lock_type(size_t tet_id, AMR::Edge_Lock_Case check, AMR::Edge_Lock_Case new_case);
            void remove_edge_locks(int intermediate = 0);
            void remove_normals();

            size_t convert_derefine_edges_to_points(
                    size_t num_edges_to_derefine,
                    AMR::Refinement_Case  refinement_case);

            std::unordered_set<size_t> child_exclusive_nodes(size_t tet_id);

    };
}

#endif //QUINOA_MESH_ADAPTER_H
