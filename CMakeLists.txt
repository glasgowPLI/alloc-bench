cmake_minimum_required(VERSION 3.15.0)
project(cheri-bdwgc 
        VERSION 0.1
        DESCRIPTION "mimalloc benchmarks"
        LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
        message(STATUS "No build type selected, default to *** Release with Debug Info ***")
        set(CMAKE_BUILD_TYPE "Debug")
endif()

# Default to bdwgc - for testing purposes
#if (NOT gclib)
#    set(gclib bdwgc)
#endif()

if (${gclib} STREQUAL bdwgc)
    set(bdwgc_cmake_args 
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/bdwgc
        -Denable_gcj_support=OFF 
        -Denable_parallel_mark=OFF 
	-Denable_threads=OFF
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
    set(bdwgc_build_prefix ${CMAKE_CURRENT_BINARY_DIR}/bdwgc)

    if (DEFINED CMAKE_TOOLCHAIN_FILE)
        message("-- Use toolchain: " ${CMAKE_TOOLCHAIN_FILE})
        list(APPEND bdwgc_cmake_args -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    else()
        message("Information:  Using HOST toolchain")
    endif()

    if (DEFINED BUILD_SHARED_LIBS)
        list(APPEND bdwgc_cmake_args -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS})
    endif()

    if (DEFINED BENCHLIB)
        list(APPEND bdwgc_cmake_args -Denable_dynamic_loading=${BENCHLIB})
    else()
        list(APPEND bdwgc_cmake_args -Denable_dynamic_loading=OFF)
    endif()
    
    message("Information:  cmake args = " ${bdwgc_cmake_args})

    # Cannot use Ninja with this due to strict external library checking
    # Will work with Unix Make
    include(ExternalProject)
    ExternalProject_Add(project_bdwgc
            GIT_REPOSITORY https://github.com/djichthys/bdwgc.git
            GIT_TAG cheri/may2025-bmabi
            CMAKE_ARGS ${bdwgc_cmake_args}
            PREFIX ${bdwgc_build_prefix})

    ExternalProject_Get_property(project_bdwgc install_dir)
    ExternalProject_Get_property(project_bdwgc source_dir)

    if (BUILD_SHARED_LIBS STREQUAL "OFF")
      add_library(gc STATIC IMPORTED)
      set_target_properties(gc PROPERTIES IMPORTED_LOCATION ${install_dir}/lib/libgc.a)
      add_dependencies(gc project_bdwgc)

      add_library(cord STATIC IMPORTED)
      set_target_properties(cord PROPERTIES IMPORTED_LOCATION ${install_dir}/lib/libcord.a)
      add_dependencies(cord project_bdwgc)

      # These libraries are built using ExternalProject. install(TARGETS) does not work
      #file(GLOB gclibraries FOLLOW_SYMLINKS ${bdwgc_build_prefix}/lib/lib*)
      set(gclibraries libgc.a libcord.a)
    else() 
      add_library(gc SHARED IMPORTED)
      set_target_properties(gc PROPERTIES IMPORTED_LOCATION ${install_dir}/lib/libgc.so)
      add_dependencies(gc project_bdwgc)

      add_library(cord SHARED IMPORTED)
      set_target_properties(cord PROPERTIES IMPORTED_LOCATION ${install_dir}/lib/libcord.so)
      add_dependencies(cord project_bdwgc)

      # These libraries are built using ExternalProject. install(TARGETS) does not work
      #file(GLOB gclibraries FOLLOW_SYMLINKS ${bdwgc_build_prefix}/lib/lib*)
      set(gclibraries 
              libgc.so.1.5.3 libgc.so.1 libgc.so
              libcord.so.1.5.1 libcord.so.1 libcord.so)
    endif() 

    list(TRANSFORM gclibraries PREPEND ${bdwgc_build_prefix}/lib/)
    install(FILES 
        ${gclibraries}
        DESTINATION lib
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
elseif(${gclib} STREQUAL snmalloc)
    set(snmalloc_cmake_args 
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/snmalloc
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
    set(snmalloc_build_prefix ${CMAKE_CURRENT_BINARY_DIR}/snmalloc)

    if (DEFINED CMAKE_TOOLCHAIN_FILE)
        message("-- Use toolchain: " ${CMAKE_TOOLCHAIN_FILE})
        list(APPEND snmalloc_cmake_args -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    else()
        message("Information:  Using HOST toolchain")
    endif()


    if (DEFINED BUILD_SHARED_LIBS)
        list(APPEND snmalloc_cmake_args -DSNMALLOC_HEADER_ONLY_LIBRARY=${BUILD_SHARED_LIBS})
    endif()

    # Cannot use Ninja with this due to strict external library checking
    # Will work with Unix Make
    include(ExternalProject)
    ExternalProject_Add(project_snmalloc
            GIT_REPOSITORY https://github.com/microsoft/snmalloc
            GIT_TAG 0a5eb403adf85b3e04ad90e6b44e0cb1de99ba38
            CMAKE_ARGS ${snmalloc_cmake_args}
            PREFIX ${snmalloc_build_prefix})

    ExternalProject_Get_property(project_snmalloc install_dir)
    ExternalProject_Get_property(project_snmalloc source_dir)

    if (BUILD_SHARED_LIBS STREQUAL "OFF")
        message(WARNING "snmalloc BUILD_SHARED_LIBS = " False)
    else() 
        message(WARNING "snmalloc BUILD_SHARED_LIBS = " True)
    endif()
    message(WARNING "snmalloc dir setup = " ${snmalloc_build_prefix})
elseif(${gclib} STREQUAL cheribumpalloc)
    add_subdirectory(cheri_bump_alloc) 
endif()

# Each individual test has own subdir
add_subdirectory(barnes) 
add_subdirectory(binary_tree) 
add_subdirectory(cfrac) 
add_subdirectory(espresso) 
add_subdirectory(richards) 
add_subdirectory(simple_alloc) 
#add_subdirectory(glibc-bench) 
add_subdirectory(binary-tree-profiling) ## TODO this
if ( NOT ${gclib} STREQUAL bdwgc)
    add_subdirectory(mstress) 
    add_subdirectory(rptest) 
    add_subdirectory(xmalloc) 
endif()

# Check out this
# [link](https://www.jwlawson.co.uk/interest/2020/02/23/cmake-external-project.html) 
# for embedding the external libgc and installation
