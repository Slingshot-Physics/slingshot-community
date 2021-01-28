set(
   executables
   # test_segment.cu
   test_plop.cu
   test_vector3.cu
)

set(
   targets
   # cuda_segment_main
   cuda_plop_main
   cuda_vec3_main
)

foreach (exec target IN ZIP_LISTS executables targets)

   add_executable(
      ${target}
      ${exec}
   )
   
   target_link_libraries(
      ${target}
      edbdgeometry
      edbdcudageometry
      edbdcudamath
      edbdcudabuffers
      plop
   )

   set_target_properties(
      ${target}
      PROPERTIES
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
   )

endforeach()
