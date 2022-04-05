cmake_minimum_required(VERSION 3.12)

project(ISL VERSION 0.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
	message(
		FATAL_ERROR
		"CMake generation for ISL is not allowed
		within the source directory!"
		)
endif()

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)


include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY    https://github.com/pybind/pybind11
	GIT_TAG           v2.6.2
	GIT_PROGRESS TRUE
	SOURCE_DIR  "${PROJECT_SOURCE_DIR}/cpp/external/pybind11"
	CONFIGURE_COMMAND   ""
	BUILD_COMMAND       ""
	INSTALL_COMMAND     ""
	TEST_COMMAND        ""
	)
FetchContent_MakeAvailable(pybind11)

find_package (Eigen3 3.3 NO_MODULE)

if (NOT TARGET Eigen3::Eigen)
	message("Could not find Eigen3 on your system, getting it myself, please wait")

	# Unfortunately, FetchContent does not work nicely. It does
	# not includ ethe target Eigen3::Eigen and it pollutes the
	# project file with many Eigen targets that is not needed.
	#include(FetchContent)
	#FetchContent_Declare(
	#	eigen
	#	GIT_REPOSITORY    https://gitlab.com/libeigen/eigen.git
	#	GIT_TAG                 3.3.9
	#	GIT_PROGRESS TRUE
	#	SOURCE_DIR  "${PROJECT_SOURCE_DIR}/cpp/external/Eigen3"
	#	CONFIGURE_COMMAND   ""
	#	BUILD_COMMAND       ""
	#	INSTALL_COMMAND     ""
	#	TEST_COMMAND        ""
	#	)
	#set(BUILD_TESTING OFF CACHE BOOL "")
	#set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
  	#set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "")
	#FetchContent_MakeAvailable(eigen)
	#set(BUILD_TESTING OFF CACHE BOOL "")
	#set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
  	#set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "")
	#FetchContent_MakeAvailable(eigen)

	include(ExternalProject)
	ExternalProject_Add(
		eigen
		GIT_REPOSITORY    https://gitlab.com/libeigen/eigen.git
		GIT_TAG                 3.3.9
		GIT_PROGRESS TRUE
		SOURCE_DIR  "${PROJECT_SOURCE_DIR}/cpp/external/Eigen3"
		CONFIGURE_COMMAND   ""
		BUILD_COMMAND       ""
		INSTALL_COMMAND     ""
		TEST_COMMAND        ""
		)	
	add_library(libisl_eigen INTERFACE)
	target_include_directories(libisl_eigen SYSTEM INTERFACE
		$<BUILD_INTERFACE: ${PROJECT_SOURCE_DIR}/cpp/external/Eigen3/>
		$<INSTALL_INTERFACE:include>
		)
	set_property(TARGET libisl_eigen PROPERTY EXPORT_NAME Eigen3::Eigen)
	add_library(Eigen3::Eigen ALIAS libisl_eigen)
	
endif ()


add_subdirectory(cpp)
