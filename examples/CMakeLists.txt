file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/examples/*.cpp")

foreach(source_file ${examples_srcs})
  get_filename_component(name ${source_file} NAME_WE) # filename without directory
  get_filename_component(path ${source_file} PATH) # directory without filename
  get_filename_component(folder ${path} NAME_WE) # example name (e.g. cmu)
  
  add_executable(${name} ${source_file})
  
  #target_link_libraries(${name} PUBLIC ${wasabi2_LINKER_LIBS} wasabi2)
  target_link_libraries(${name} PUBLIC wasabi2)
  #target_include_directories(${name} PUBLIC ${wasabi2_INCLUDE_DIRS} ${wasabi2_INCLUDE_DIR}) 
  #target_include_directories(${name} PUBLIC ${wasabi2_INCLUDE_DIRS}) 

  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/examples/")
  install(TARGETS ${name} DESTINATION ${PROJECT_BINARY_DIR}/examples/)
endforeach()
