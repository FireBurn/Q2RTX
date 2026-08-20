find_program(SEVEN_ZIP_EXECUTABLE NAMES 7zz 7z 7za REQUIRED)

function(package_pkz output_file)
    execute_process(
        COMMAND "${SEVEN_ZIP_EXECUTABLE}" a -tzip "${output_file}" ${ARGN}
        RESULT_VARIABLE package_result
    )
    if(NOT package_result EQUAL 0)
        message(FATAL_ERROR
            "7-Zip failed (${package_result}) while creating ${output_file}")
    endif()
endfunction()

# The source checkout intentionally excludes NVIDIA's large media trees.  The
# Linux release archive supplies q2rtx_media.pkz; replace only the small,
# source-controlled configuration entries so an installed live build gets the
# current menu without dropping the released textures/models/audio.
set(out_file "${SOURCE}/baseq2/q2rtx_media.pkz")
if(NOT EXISTS "${out_file}")
    message(FATAL_ERROR "Missing released media archive: ${out_file}")
endif()
execute_process(
    # `a`, rather than `u`, deliberately replaces all four files even when a
    # source checkout preserved an older timestamp than the release archive.
    COMMAND "${SEVEN_ZIP_EXECUTABLE}" a -tzip "${out_file}"
        prefetch.txt pt_toggles.cfg q2rtx.cfg q2rtx.menu
    WORKING_DIRECTORY "${SOURCE}/baseq2"
    RESULT_VARIABLE package_result
)
if(NOT package_result EQUAL 0)
    message(FATAL_ERROR
        "7-Zip failed (${package_result}) while updating ${out_file}")
endif()

SET(MEDIA_SOURCES_ROGUE
    ${SOURCE}/rogue/maps
)
set(out_file_rogue "${SOURCE}/rogue/q2rtx_media.pkz")
package_pkz("${out_file_rogue}" ${MEDIA_SOURCES_ROGUE})
