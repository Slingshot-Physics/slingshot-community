add_library(
   transform_ui
   SHARED
   transform_ui.cpp
   transform_ui.hpp
)

target_link_libraries(
   transform_ui
   physics
   test_utils
   viz
)

set(
   lib_targets
   line_loader_ui
   aabb_ui
   polygon_ui
   polyhedron_loader_ui
   point_cloud_ui
   shape_loader_ui
   tetrahedron_ui
   triangle_ui
)

foreach( lib_target ${lib_targets} )

   set( source ${lib_target}.cpp)

   set( header )

   if( EXISTS ${lib_target}.hpp)
      set( header ${lib_target}.hpp)
   endif()

   add_library(
      ${lib_target}
      SHARED
      ${source}
      ${header}
   )

   target_link_libraries(
      ${lib_target}
      physics
      test_utils
      transform_ui
      viz
   )

endforeach()

set(
   exec_targets
   aabb_tree_ui
   gjk_epa_ui
   gjk_epa_verify_ui
   graham_scan_ui
   mesh_feature_ui
   make_gauss_maps
   polygon_segment_ui
   quickhull_ui
   ray_intersection_ui
   ray_shape_intersection_ui
   sat_collision_ui
   sat_verify_ui
   segment_aabb_ui
   segment_line_ui
   segment_segment_ui
   shape_feature_isometric_ui
   shape_feature_ui
   shape_polyhedron_collision_ui
   shape_shape_collision_ui
   sutherland_hodgman_ui
   tetrahedron_closest_point_ui
   triangle_closest_point_ui
   typedef_test
)

foreach( exec_target ${exec_targets} )

   set( source ${exec_target}.cpp)

   set( header )

   if( EXISTS ${exec_target}.hpp)
      set( header ${exec_target}.hpp)
   endif()

   add_executable(
      ${exec_target}
      ${source}
      ${header}
   )

   target_link_libraries(
      ${exec_target}
      physics
      viz
      ${lib_targets}
   )

endforeach()
