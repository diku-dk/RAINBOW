pybind11_add_module(
	pyisl
	py_isl_tetgen.cpp
	)

target_link_libraries(
	pyisl
	PRIVATE
	isl_tetgen
	)

install(
	TARGETS pyisl
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include
	)

set_target_properties(
	pyisl
	PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY_DEBUG    ${PROJECT_SOURCE_DIR}/python
	LIBRARY_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/python
	)
