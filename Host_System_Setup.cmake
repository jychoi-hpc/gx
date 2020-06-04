SET (AVAILABLE_SYSTEMS gx)
CMAKE_HOST_SYSTEM_INFORMATION (RESULT SYSTEM_NAME QUERY HOSTNAME)

FOREACH (AVAILABLE_SYSTEM_ID ${AVAILABLE_SYSTEMS})
  STRING (FIND ${SYSTEM_NAME} ${AVAILABLE_SYSTEM_ID} STRING_LOCATION)
  IF (NOT ${STRING_LOCATION} EQUAL -1)
    #MESSAGE ("************************************************************")          
    #MESSAGE ("CMake has detected ${AVAILABLE_SYSTEM_ID} as the system host")          
    SET (HOST_SYSTEM ${AVAILABLE_SYSTEM_ID})
    BREAK ()
  ENDIF ()
ENDFOREACH ()

IF (NOT HOST_SYSTEM)
  MESSAGE (FATAL_ERROR "CMake was not able to detect a known host system. Exiting...")
ENDIF ()

#---- "Allocating" empty vector to be filled with -D flags depending on which packages are used
SET (INCLUDE_LIST "")
SET (COMPILE_DEF_LIST "")
SET (LIBRARY_LINK_LIST "")
SET (NVCC_FLAGS "")

IF (${HOST_SYSTEM} MATCHES gx)

  MESSAGE ("")
  MESSAGE ("")
  MESSAGE ("=========================================================")
  MESSAGE ("")
  MESSAGE ("GX selected as host system")
  MESSAGE ("")
  MESSAGE ("=========================================================")

  FIND_PACKAGE(MPI)
  IF (MPI_FOUND)
    INCLUDE_DIRECTORIES (SYSTEM ${MPI_CXX_INCLUDE_PATH})
    LIST (APPEND LIBRARY_LINK_LIST ${MPI_CXX_LIBRARIES})
  ENDIF ()

  SET (CUDAARCH 70)
  SET (CUDA_INC /usr/local/cuda-10.2/targets/x86_64-linux/include)

  LIST (APPEND NVCC_FLAGS -dc)
  LIST (APPEND NVCC_FLAGS -arch=compute_${CUDAARCH})
  LIST (APPEND NVCC_FLAGS -code=sm_${CUDAARCH})
  LIST (APPEND NVCC_FLAGS -use_fast_math)
  LIST (APPEND NVCC_FLAGS -prec-sqrt=true)
  
ENDIF ()

