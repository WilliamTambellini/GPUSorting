cmake_minimum_required (VERSION 2.9.3)
project (GPUComputing)

# Add our modules to the path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/")

include(CheckCXXCompilerFlag)

#set (EXTRA_COMPILE_FLAGS "-Wall -Werror")
set (EXTRA_COMPILE_FLAGS "-Wall")
CHECK_CXX_COMPILER_FLAG(-std=c++11 HAS_CXX_11)
CHECK_CXX_COMPILER_FLAG(-std=c++14 HAS_CXX_14)
if (HAS_CXX_14)
    set(EXTRA_COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -std=c++14")
    message(STATUS "Enabling C++14 support")
else()
    message(FATAL "No C++14 support detected.")
endif()

message(STATUS ${EXTRA_COMPILE_FLAGS})
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${EXTRA_COMPILE_FLAGS}")

# Include support for changing the working directory in Visual Studio
include(ChangeWorkingDirectory)

# Search for OpenCL and add paths
find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

# Include Common module
add_subdirectory (../Common ${CMAKE_BINARY_DIR}/Common)

# Define source files for this assignment
FILE(GLOB Sources *.cpp)
FILE(GLOB Headers *.h)
FILE(GLOB CLSources *.cl)
ADD_EXECUTABLE (Sorting
	${Sources}
	${Headers}
	${CLSources}
)

# Link required libraries
target_link_libraries(Sorting m stdc++ ${OPENCL_LIBRARIES})
target_link_libraries(Sorting GPUCommon)
