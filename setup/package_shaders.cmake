SET(SHADER_SOURCES
    ${SOURCE}/baseq2/shader_vkpt
)
set(out_file "${SOURCE}/baseq2/shaders.pkz")
find_program(SEVEN_ZIP_EXECUTABLE NAMES 7zz 7z 7za REQUIRED)
execute_process(
    COMMAND "${SEVEN_ZIP_EXECUTABLE}" a -tzip "${out_file}" ${SHADER_SOURCES}
    RESULT_VARIABLE package_result
)
if(NOT package_result EQUAL 0)
    message(FATAL_ERROR
        "7-Zip failed (${package_result}) while creating ${out_file}")
endif()
