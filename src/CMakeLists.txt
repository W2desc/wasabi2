file(GLOB_RECURSE srcs "${PROJECT_SOURCE_DIR}/src/*.cpp")
#message("wasabi2_INCLUDE_DIR: ${wasabi2_INCLUDE_DIR}")
#message("wasabi2_INCLUDE_DIRS: ${wasabi2_INCLUDE_DIRS}")
#message("wasabi2_LINKER_LIBS: ${wasabi2_LINKER_LIBS}")

add_library(wasabi2 ${srcs})
target_link_libraries(wasabi2 ${wasabi2_LINKER_LIBS})
target_include_directories(wasabi2 PUBLIC ${wasabi2_INCLUDE_DIRS} ${wasabi2_INCLUDE_DIR})
