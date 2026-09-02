SET(SHADER_SOURCES
    ${SOURCE}/baseq2/shader_vkpt
)
set(out_file "${SOURCE}/baseq2/shaders.pkz")
find_program(SEVEN_ZIP_EXECUTABLE NAMES 7zz 7z 7za REQUIRED)
# Do not retain obsolete entries from a prior descriptor-layout generation.
# `7z a` only updates matching names; recreating the archive makes its ABI
# exactly match the shader_vkpt directory verified by the build target.
file(REMOVE "${out_file}")
execute_process(
    COMMAND "${SEVEN_ZIP_EXECUTABLE}" a -tzip "${out_file}" ${SHADER_SOURCES}
    RESULT_VARIABLE package_result
)
if(NOT package_result EQUAL 0)
    message(FATAL_ERROR
        "7-Zip failed (${package_result}) while creating ${out_file}")
endif()
