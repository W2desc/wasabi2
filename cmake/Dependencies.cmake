
set(wasabi2_LINKER_LIBS "")
set(wasabi2_INCLUDE_DIRS "")

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs features2d)
#if(NOT OpenCV_FOUND) 
#  find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
#endif()

list(APPEND wasabi2_INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
list(APPEND wasabi2_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
list(APPEND wasabi2_DEFINITIONS PUBLIC -DUSE_OPENCV)
