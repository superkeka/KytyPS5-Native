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
