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

# Shader scanner test: synthetic gfx10 words only (no title's shader bytes), CPU only, so it runs on the dev
# machine too. shaderScanNative.cpp (shader.cpp's registry, xxHash) is the emulator's and is not part of it.
add_executable(native_gpu_shader_scan_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/NativeGpuShaderScanTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/native_gpu/shaderScan.cpp"
	"${KYTY_SOURCE_DIR}/graphics/native_gpu/shaderPatcher.cpp"
)
target_include_directories(native_gpu_shader_scan_tests PRIVATE ${inc_headers})

if(BUILD_TESTING)
	add_test(NAME native_gpu_shader_scan COMMAND $<TARGET_FILE:native_gpu_shader_scan_tests>)
	add_dependencies(kyty_tests native_gpu_shader_scan_tests)
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

# Hang watchdog decisions (nativeWatch.h): header-only and pure, so it runs everywhere.
add_executable(native_gpu_watch_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/NativeGpuWatchTests.cpp"
)
target_include_directories(native_gpu_watch_tests PRIVATE ${inc_headers})

if(BUILD_TESTING)
	add_test(NAME native_gpu_watch COMMAND $<TARGET_FILE:native_gpu_watch_tests>)
	add_dependencies(kyty_tests native_gpu_watch_tests)
endif()

# The root helper (tools/bc250-lodhw): the privileged registers the native backend needs (LOD
# statistics, KFD TLB flushes). Built with the emulator and shipped next to it with its service
# and installer (sudo ./install-helper.sh).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	add_executable(kyty-lodhw "${CMAKE_CURRENT_SOURCE_DIR}/tools/bc250-lodhw/kyty-lodhw.c")
	target_link_libraries(kyty-lodhw PRIVATE Threads::Threads)
	cmake_language(DEFER CALL add_dependencies kyty_emulator kyty-lodhw)
	install(TARGETS kyty-lodhw DESTINATION .)
	install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tools/bc250-lodhw/install-helper.sh" DESTINATION .)
	install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tools/bc250-lodhw/kyty-lodhw.service" DESTINATION .)
	# What the Steam shortcuts (the Decky plugin adds them) start: unloads Steam, boot and exit videos.
	install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tools/kytyps5-native/kytyps5-run.sh" DESTINATION .)
endif()
