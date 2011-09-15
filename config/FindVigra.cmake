# This module finds an installed Vigra package.
#
# VIGRA_FOUND
# VIGRA_INCLUDE_DIR
# VIGRA_NUMPY_MODULE_DIR
# VIGRA_NUMPY_CORE_LIBRARY
# VIGRA_NUMPY_IMPEX_LIBRARY
# VIGRA_INCLUDE_DIR
# VIGRA_IMPEX_LIBRARY
# VIGRA_IMPEX_LIBRARY_FOUND

#see:
#http://www.mail-archive.com/cmake@cmake.org/msg29402.html
find_package(PythonLibs REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(Numpy REQUIRED)
find_package(PackageHandleStandardArgs REQUIRED)

function(find_python_module module)
    string(TOUPPER ${module} module_upper)
        if(NOT PY_${module_upper})
            if(ARGC GREATER 1 AND ARGV1 STREQUAL "REQUIRED")
                set(${module}_FIND_REQUIRED TRUE)
            endif()
            # A module's location is usually a directory, but for binary modules
            # it's a .so file.
            set(cmd "import re, ${module}\; print re.compile('/__init__.py.*').sub('',${module}.__file__)")
            execute_process(COMMAND ${PYTHON_EXECUTABLE} "-c" ${cmd}
                            RESULT_VARIABLE _${module}_status 
                            OUTPUT_VARIABLE _${module}_location
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT _${module}_status)
                set(PY_${module_upper} "${_${module}_location}"
			        CACHE STRING 
                    "Location of Python module ${module}")
            endif()
        endif()
        find_package_handle_standard_args(PY_${module} DEFAULT_MSG PY_${module_upper})
endfunction()

FIND_PATH(VIGRA_INCLUDE_DIR vigra/matrix.hxx)
FIND_LIBRARY(VIGRA_IMPEX_LIBRARY NAMES vigraimpex)

find_python_module(vigra REQUIRED)
SET(VIGRA_NUMPY_MODULE_DIR ${PY_VIGRA})

FIND_FILE(VIGRA_NUMPY_CORE_LIBRARY
          NAMES vigranumpycore.so
          PATHS "${PY_VIGRA}" NO_CMAKE_SYSTEM_PATH NO_CMAKE_PATH NO_DEFAULT_PATH)
FIND_FILE(VIGRA_NUMPY_IMPEX_LIBRARY
          NAMES impex.so
          PATHS "${PY_VIGRA}" NO_CMAKE_SYSTEM_PATH NO_CMAKE_PATH NO_DEFAULT_PATH)

IF (VIGRA_INCLUDE_DIR)
    SET(VIGRA_FOUND TRUE)
ENDIF()

IF(VIGRA_IMPEX_LIBRARY)
    SET(VIGRA_IMPEX_LIBRARY_FOUND TRUE)
ENDIF()

IF(VIGRA_FOUND)
    IF (NOT Vigra_FIND_QUIETLY)
      MESSAGE(STATUS "Found Vigra:")
      MESSAGE(STATUS "  > includes:            ${VIGRA_INCLUDE_DIR}")
      MESSAGE(STATUS "  > impex library:       ${VIGRA_IMPEX_LIBRARY}")
      MESSAGE(STATUS "  > numpy core library:  ${VIGRA_NUMPY_CORE_LIBRARY}")
      MESSAGE(STATUS "  > numpy impex library: ${VIGRA_NUMPY_IMPEX_LIBRARY}")

    ENDIF()
ELSE (VIGRA_FOUND)
    IF(Vigra_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find Vigra")
    ENDIF()
ENDIF()
