project(priows)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/config)

find_package(Numpy REQUIRED)
find_package(Vigra REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python)

include_directories(
    ${VIGRA_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${PYTHON_NUMPY_INCLUDE_DIR}
)

add_library(priows SHARED
            python_svs.cpp)
target_link_libraries(priows ${Boost_LIBRARIES} ${VIGRA_NUMPY_CORE_LIBRARY} gomp)
IF(WIN32)
    SET_TARGET_PROPERTIES(priows PROPERTIES OUTPUT_NAME "priows" PREFIX "" SUFFIX  ".pyd")
ELSEIF(MACOSX)
    SET_TARGET_PROPERTIES(priows PROPERTIES OUTPUT_NAME "priows" PREFIX "" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(priows PROPERTIES OUTPUT_NAME "priows" PREFIX "")
ENDIF()

install(TARGETS priows LIBRARY DESTINATION ${VIGRA_NUMPY_MODULE_DIR})
              
      

