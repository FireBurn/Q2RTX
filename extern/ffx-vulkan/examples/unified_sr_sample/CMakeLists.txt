cmake_minimum_required(VERSION 3.15)
project(unified_sr_minimal_sample LANGUAGES C)

find_package(Vulkan REQUIRED)

add_executable(unified_sr_minimal_sample unified_sr_minimal_sample.c)
target_compile_features(unified_sr_minimal_sample PRIVATE c_std_11)
target_link_libraries(unified_sr_minimal_sample PRIVATE ffx-vulkan::unified-sr)

if(MSVC)
    target_compile_options(unified_sr_minimal_sample PRIVATE /W4)
else()
    target_compile_options(unified_sr_minimal_sample PRIVATE -Wall -Wextra -Wpedantic)
endif()
