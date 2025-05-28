# CMakeList.txt: проект CMake для CMakeProject1; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)
project("Z-spline")

find_package (Python3 COMPONENTS Interpreter Development REQUIRED)
#add_executable(zspline1 pywrap.cpp)
execute_process(
	COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
	OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(zspline SHARED pywrap.cpp)

target_link_libraries(zspline Eigen3::Eigen)
set_target_properties(zspline PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

set(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
if(NOT EXISTS ${INIT_PY})
file(WRITE ${INIT_PY} "")
endif()

add_custom_command(TARGET zspline POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
$<TARGET_FILE_DIR:zspline>/$<TARGET_FILE_NAME:zspline>
${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:zspline>
)

#add_executable (pywrap pywrap.cpp)
#if (CMAKE_VERSION VERSION_GREATER 3.12)
#  set_property(TARGET CMakeProject1 PROPERTY CXX_STANDARD 20)
#endif()
#find_package(Eigen3 REQUIRED)
#target_link_libraries(CMakeProject1 Eigen3::Eigen)
#add_subdirectory(pybind11)
#find_package(pybind11 REQUIRED)
#pybind11_add_module(Z-spline SHARED pywrap.cpp)

