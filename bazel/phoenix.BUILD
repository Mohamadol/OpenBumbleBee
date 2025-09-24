load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

_PHX_CACHE = {
    "CMAKE_CXX_STANDARD": "17",
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
    "BUILD_SHARED_LIBS": "OFF",
    "CMAKE_INSTALL_LIBDIR": "lib",

    "SEAL_DIR": "$EXT_BUILD_DEPS/SEAL-4.1/lib/cmake/SEAL",
    # "Eigen3_DIR": "$EXT_BUILD_DEPS/eigen3/share/eigen3/cmake",
    "HEXL_DIR": "$EXT_BUILD_DEPS/hexl/lib/cmake",
    "CpuFeatures_DIR": "$EXT_BUILD_DEPS/cpu_features/lib/cmake/CpuFeatures",
    "ZSTD_DIR": "$EXT_BUILD_DEPS/zstd/lib/cmake/zstd",
    "EXT_BUILD_DEPS": "$EXT_BUILD_DEPS",

    # Phoenix-specific toggles if applicable (harmless if unused)
    "PHOENIX_BUILD_TESTS": "OFF",
    "PHOENIX_BUILD_EXAMPLES": "OFF",

    # Let Phoenix CMake use prebuilt deps
    "BUILD_DEPS": "OFF",
    "BUILD_FOR_BAZEL": "ON",
}

spu_cmake_external(
    name = "phoenix",
    lib_source = "@com_github_phoenix//:all_srcs",
    cache_entries = _PHX_CACHE,
    out_include_dir =  "include",
    out_static_libs = [
        "libphoenix.a",
        "libphoenix_core.a",
        "libphoenix_comm.a",
    ],
    deps = [
        "@com_github_eigenteam_eigen//:eigen3",
        "@com_github_microsoft_seal//:seal",
    ],
    copts = [
        "-maes",
        "-msse4.1",
        "-mpclmul",
        "-DHAVE_AESNI=1",
    ],
    defines = ["HAVE_AESNI=1"], 
    linkopts = ["-lm"], 
)
