target_sources(${amr_wind_exe_name}
   PRIVATE
      #C++
      sgs_model.cpp
			#Headers
			sgs_model.H
   )

#target_include_directories(${amr_wind_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
