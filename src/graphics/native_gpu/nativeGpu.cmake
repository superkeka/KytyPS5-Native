# BC-250 native GPU backend (fork-only). Included once from the top-level CMakeLists.txt so the
# fork touches upstream build files in a single line.

add_executable(native_gpu_shader_patcher_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/NativeGpuShaderPatcherTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/native_gpu/shaderPatcher.cpp"
)
target_include_directories(native_gpu_shader_patcher_tests PRIVATE ${inc_headers})

if(BUILD_TESTING)
	add_test(NAME native_gpu_shader_patcher COMMAND $<TARGET_FILE:native_gpu_shader_patcher_tests>)
	add_dependencies(kyty_tests native_gpu_shader_patcher_tests)
endif()

# GEM backing is header-only (src/graphics/native_gpu/gemBacking.h), so the emulator needs no extra
# source; this standalone test exercises it directly and skips cleanly off the board.
add_executable(native_gpu_gem_backing_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/NativeGpuGemBackingTests.cpp"
)
target_include_directories(native_gpu_gem_backing_tests PRIVATE ${inc_headers})

if(BUILD_TESTING)
	add_test(NAME native_gpu_gem_backing COMMAND $<TARGET_FILE:native_gpu_gem_backing_tests>)
	add_dependencies(kyty_tests native_gpu_gem_backing_tests)
endif()

# PM4 translator test: pure data transform (no hardware), so it runs everywhere including the dev
# machine.
add_executable(native_gpu_pm4_translator_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/NativeGpuPm4TranslatorTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/native_gpu/nativePm4.cpp"
)
target_include_directories(native_gpu_pm4_translator_tests PRIVATE ${inc_headers})

if(BUILD_TESTING)
	add_test(NAME native_gpu_pm4_translator COMMAND $<TARGET_FILE:native_gpu_pm4_translator_tests>)
	add_dependencies(kyty_tests native_gpu_pm4_translator_tests)
endif()

# Frame capture format and PM4 helpers (nativeCapture.h): portable, runs everywhere.
add_executable(native_gpu_capture_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/NativeGpuCaptureTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
)
target_include_directories(native_gpu_capture_tests PRIVATE ${inc_headers})

if(BUILD_TESTING)
	add_test(NAME native_gpu_capture COMMAND $<TARGET_FILE:native_gpu_capture_tests>)
	add_dependencies(kyty_tests native_gpu_capture_tests)
endif()

# Replays a KYTY_NATIVE_GPU_CAPTURE frame capture on the BC-250 (tools/bc250-replay).
if(LINUX)
	add_executable(bc250_replay
		"${CMAKE_CURRENT_SOURCE_DIR}/tools/bc250-replay/replay.cpp"
		"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	)
	target_include_directories(bc250_replay PRIVATE ${inc_headers} "${CMAKE_CURRENT_SOURCE_DIR}/tools/bc250-replay")
	target_link_libraries(bc250_replay PRIVATE Threads::Threads)
endif()
