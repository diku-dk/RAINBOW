add_library(
  tetgen
  predicates.cxx
  tetgen.cxx
  tetgen.h
  )

set_target_properties(
  tetgen
  PROPERTIES
  PUBLIC_HEADER "tetgen.h"
  )

target_include_directories(
  tetgen
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/external/tetgen>
  $<INSTALL_INTERFACE:include>
  )

target_compile_definitions(
  tetgen
  PUBLIC
  TETLIBRARY
  )

install(
  TARGETS tetgen
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
  )

set_target_properties(
  tetgen
	PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/python"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/python"
	)
