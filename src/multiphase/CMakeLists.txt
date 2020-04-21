target_sources(${amr_wind_lib_name}
  PRIVATE
    #Source
    VolumeFractionUtils.cpp
    #Headers
    VolumeFraction_K.H
  )

target_include_directories(
  ${amr_wind_lib_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
