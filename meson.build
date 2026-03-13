project('faac', 'c', version: '1.31.1', default_options: ['default_library=both'])

add_global_arguments('-DHAVE_CONFIG_H', language: 'c')

config_h = configuration_data()
config_h.set_quoted('PACKAGE', meson.project_name())
config_h.set_quoted('PACKAGE_VERSION', meson.project_version())

cc = meson.get_compiler('c')
foreach h: ['getopt.h', 'immintrin.h', 'stdint.h', 'sys/time.h', 'sys/types.h']
    config_h.set('HAVE_' + h.underscorify().to_upper(), cc.has_header(h))
endforeach


config_h.set('HAVE_STRCASECMP', cc.has_function('strcasestr'))


libm = cc.find_library('m', required: false)

cpu_family = target_machine.cpu_family()

# SIMD: higher levels require lower levels for remainder handling
# AVX512 -> AVX2, SSE2; AVX2 -> SSE2
simd_base = (cpu_family in ['x86', 'x86_64']) and config_h.get('HAVE_IMMINTRIN_H')
have_avx512 = simd_base
have_avx2 = simd_base or have_avx512
have_sse2 = simd_base or have_avx2 or have_avx512
config_h.set('HAVE_SSE2', have_sse2)
config_h.set('HAVE_AVX2', have_avx2)
config_h.set('HAVE_AVX512', have_avx512)

if get_option('floating-point') == 'double'
    config_h.set('FAAC_PRECISION_DOUBLE', 1)
else
    config_h.set('FAAC_PRECISION_SINGLE', 1)
endif

c_args = []
if cc.get_argument_syntax() == 'msvc'
    c_args += ['-DWIN32', '-DNDEBUG', '-D_WINDOWS', '-D_USRDLL', '-DLIBFAAC_DLL_EXPORTS']
endif

configure_file(
    output: 'config.h',
    configuration: config_h
)


subdir('docs')
subdir('include')
subdir('libfaac')

if get_option('frontend')
    subdir('frontend')
endif
