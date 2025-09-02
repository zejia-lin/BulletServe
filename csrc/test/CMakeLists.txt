project(SMCtrl_Test)

add_library(test_common SHARED libsmctrl_test_mask_shared.cu)
target_include_directories(test_common PRIVATE 
                                      ${CUDAToolkit_INCLUDE_DIRS}
                                      ${CMAKE_SOURCE_DIR}/src)
target_link_directories(test_common PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(test_common PRIVATE smctrl cuda)
target_link_options(test_common PRIVATE LINKER:-rpath,${CUDAToolkit_LIBRARY_DIR})


file(GLOB_RECURSE source_files *.c)
foreach(source_file ${source_files})
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_directories(${target_name} PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${target_name} PRIVATE cuda smctrl test_common)
endforeach()


file(GLOB_RECURSE source_files test_*)
foreach(source_file ${source_files})
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_directories(${target_name} PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${target_name} PRIVATE cuda smctrl test_common)
endforeach()