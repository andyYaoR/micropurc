cmake_minimum_required(VERSION 3.18)
project(micropurc_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # Link the MSVC runtime statically: the extension then loads no
  # msvcp140.dll at all, so an outdated copy shadowing the system one (conda
  # environments ship their own, and the mismatch is a silent access
  # violation on import) cannot affect it. Set before any target is created
  # so it also applies to nanobind's static library.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  # Belt for source builds that keep the dynamic runtime: restore the
  # pre-14.40 std::mutex layout in every translation unit, nanobind's
  # included.
  add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# ── Python + nanobind ────────────────────────────────────────────────
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE _NB_CMAKE_DIR
    RESULT_VARIABLE _NB_RESULT)
  if(_NB_RESULT EQUAL 0 AND _NB_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${_NB_CMAKE_DIR}")
  endif()
endif()
find_package(nanobind CONFIG REQUIRED)

# ── Eigen + PIQP (header-only) ─────────────────────────────────────
# conda-forge installs Eigen's CMake config and headers under the environment
# prefix on Unix, but under <prefix>/Library on Windows. Seed CMAKE_PREFIX_PATH
# with both (from CONDA_PREFIX, which the build front-end reliably inherits) so
# find_package(Eigen3) resolves even when scikit-build-core's PEP 517 build
# isolation does not forward a CMAKE_PREFIX_PATH set in the parent shell into
# the configure subprocess.
foreach(_conda_root "$ENV{CONDA_PREFIX}" "$ENV{CONDA_PREFIX}/Library"
                    "$ENV{LIBRARY_PREFIX}" "$ENV{PREFIX}")
  if(_conda_root AND IS_DIRECTORY "${_conda_root}")
    list(APPEND CMAKE_PREFIX_PATH "${_conda_root}")
  endif()
endforeach()

# Do not pass an explicit version here. conda-forge now ships Eigen 5.x, whose
# Eigen3ConfigVersion.cmake rejects a `find_package(Eigen3 3.3)` request as
# version-incompatible (major-version mismatch) and reports NOTFOUND even when
# the config is on CMAKE_PREFIX_PATH — silently sending the build to the
# fallback below (which used to be Unix-only, hence the Windows failure). The
# code only needs a reasonably modern Eigen, which any conda-forge build
# satisfies, so we omit the version and let compilation fail loudly otherwise.
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
  # Header-only fallback: probe the conda layouts directly on both Unix
  # (<prefix>/include/eigen3) and Windows (<prefix>/Library/include/eigen3).
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense
    HINTS "$ENV{CONDA_PREFIX}" "$ENV{CONDA_PREFIX}/Library"
          "$ENV{LIBRARY_PREFIX}" "$ENV{PREFIX}"
    PATH_SUFFIXES include/eigen3 include
    PATHS "/usr/local" "/usr")
  if(EIGEN3_INCLUDE_DIR)
    message(STATUS "Found Eigen3 headers at ${EIGEN3_INCLUDE_DIR}")
  else()
    # Last resort, for clean CI runners with no conda or system Eigen (e.g.
    # cibuildwheel's manylinux containers and the GitHub macOS/Windows
    # runners): fetch a pinned Eigen release from the official GitLab mirror.
    # Eigen is header-only, so we only populate the sources and point the
    # include path at them -- Eigen's own CMake project (docs, tests, install
    # rules) is never configured, so nothing of Eigen's leaks into our build
    # or install (wheel) stage. The direct-argument FetchContent_Populate
    # form is used deliberately: it works on CMake 3.18 (our minimum) and,
    # unlike the single-argument form, is not deprecated by CMP0169 on
    # CMake >= 3.30.
    message(STATUS
      "Eigen3 not found via find_package or header probe; "
      "fetching Eigen 3.4.0 sources (header-only) with FetchContent")
    include(FetchContent)
    FetchContent_Populate(eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.4.0
      GIT_SHALLOW TRUE)
    set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}" CACHE PATH
        "Eigen3 headers (fetched by FetchContent)" FORCE)
    message(STATUS "Using fetched Eigen3 headers at ${EIGEN3_INCLUDE_DIR}")
  endif()
endif()

# PIQP header-only library (vendored in extern/piqp/).
set(PIQP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/extern/piqp")

# ── Build the extension ─────────────────────────────────────────────
# Threading via std::thread (no OpenMP — avoids dual-runtime conflicts
# with torch/numpy libomp on macOS).
nanobind_add_module(_native
    src/micropurc/_native/bindings.cpp
    src/micropurc/_native/activeset.cpp
    src/micropurc/_native/schur.cpp
    src/micropurc/_native/route_sampler.cpp)

# std::thread needs pthreads on Linux. On MinGW the pthread symbols come from
# winpthread, which we link *statically* in the MINGW block below so the .pyd
# carries no libwinpthread-1.dll dependency; using Threads::Threads there would
# re-introduce a dynamic winpthread reference, so it is skipped on MinGW.
find_package(Threads REQUIRED)
if(NOT MINGW)
  target_link_libraries(_native PRIVATE Threads::Threads)
endif()

# Eigen + PIQP headers.
if(Eigen3_FOUND)
  target_link_libraries(_native PRIVATE Eigen3::Eigen)
else()
  target_include_directories(_native PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
target_include_directories(_native PRIVATE ${PIQP_INCLUDE_DIR})
target_compile_definitions(_native PRIVATE HAVE_PIQP)

# ── BLAS/LAPACK (for large matrix operations) ──────────────────────
if(APPLE)
  target_link_libraries(_native PRIVATE "-framework Accelerate")
  target_compile_definitions(_native PRIVATE HAVE_LAPACK)
else()
  find_package(LAPACK QUIET)
  find_package(BLAS QUIET)
  if(LAPACK_FOUND AND BLAS_FOUND)
    target_link_libraries(_native PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    target_compile_definitions(_native PRIVATE HAVE_LAPACK)
  endif()
endif()

# ── Self-contained runtime on Windows/MinGW ─────────────────────────
# Built with MinGW-w64 (e.g. Strawberry Perl's GCC), the .pyd otherwise
# depends on libgcc_s_seh-1.dll, libstdc++-6.dll, and libwinpthread-1.dll
# (the last pulled in by std::thread). Since Python 3.8 the extension loader
# ignores PATH for a .pyd's own dependencies -- only the module's directory,
# the system directories, and os.add_dll_directory() are searched -- so adding
# the toolchain's bin to PATH does not resolve them. Statically link the GCC
# runtime and winpthread so the module has no external MinGW DLL dependencies.
# Surgical by design: BLAS/LAPACK and system DLLs stay dynamic.
if(MINGW)
  target_link_options(_native PRIVATE
    -static-libgcc -static-libstdc++
    -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic)
endif()

# Optimisation flags.
# MICROPURC_NATIVE_ARCH defaults to OFF because binaries built with
# -march=native are tied to the build machine's CPU: a wheel compiled on a
# modern CI runner would crash with SIGILL (illegal instruction) on older
# user CPUs. Source builders compiling for their own machine can opt in with
# -DMICROPURC_NATIVE_ARCH=ON; distributable wheels must leave it off.
option(MICROPURC_NATIVE_ARCH
  "Optimize for the build machine's CPU (never enable for distributable wheels)"
  OFF)
if(NOT MSVC)
  target_compile_options(_native PRIVATE -O3 -funroll-loops)
  if(MICROPURC_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
      target_compile_options(_native PRIVATE -march=native)
    endif()
  endif()
else()
  # No explicit MSVC optimisation flags needed: the wheel build configures
  # with cmake.build-type = Release (pyproject.toml), and CMake's default
  # CMAKE_CXX_FLAGS_RELEASE for MSVC already contains /O2 /Ob2 /DNDEBUG
  # (see Modules/Platform/Windows-MSVC.cmake). Adding /O2 here again would
  # be redundant and risk duplicate-flag warnings.
endif()

# Install into the micropurc package directory.
install(TARGETS _native DESTINATION micropurc)
