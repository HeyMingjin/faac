link_args = []
if host_machine.system() == 'windows' and meson.get_compiler('c').get_id() == 'gcc'
    # identifies mingw
    link_args += '-Wl,--add-stdcall-alias'
endif


quantize_simd_libs = []

# Build lower-width SIMD when higher is configured (for remainder handling)
# AVX512 needs AVX2, SSE2; AVX2 needs SSE2
if config_h.get('HAVE_SSE2') or config_h.get('HAVE_AVX2') or config_h.get('HAVE_AVX512')
    sse2_flag = (cc.get_argument_syntax() == 'msvc') ? ['/arch:SSE2'] : ['-msse2']
    quantize_sse = static_library('quantize_sse', 'quantize_sse.c',
        include_directories: ['..', '../include'],
        c_args: c_args + sse2_flag
    )
    quantize_simd_libs += quantize_sse
endif

if config_h.get('HAVE_AVX2') or config_h.get('HAVE_AVX512')
    avx2_flag = (cc.get_argument_syntax() == 'msvc') ? ['/arch:AVX2'] : ['-mavx2']
    quantize_avx2 = static_library('quantize_avx2', 'quantize_avx2.c',
        include_directories: ['..', '../include'],
        c_args: c_args + avx2_flag
    )
    quantize_simd_libs += quantize_avx2
endif

if config_h.get('HAVE_AVX512')
    avx512_flag = (cc.get_argument_syntax() == 'msvc') ? ['/arch:AVX512'] : ['-mavx512f', '-mavx512dq']
    quantize_avx512 = static_library('quantize_avx512', 'quantize_avx512.c',
        include_directories: ['..', '../include'],
        c_args: c_args + avx512_flag
    )
    quantize_simd_libs += quantize_avx512
endif

common_src = [
    'bitstream.c',
    'bitstream.h',
    'blockswitch.c',
    'blockswitch.h',
    'channels.c',
    'channels.h',
    'coder.h',
    'cpu_compute.c',
    'cpu_compute.h',
    'filtbank.c',
    'filtbank.h',
    'fft.c',
    'fft.h',
    'frame.c',
    'frame.h',
    'huff2.c',
    'huff2.h',
    'huffdata.c',
    'huffdata.h',
    'quantize.c',
    'quantize.h',
    'stereo.c',
    'stereo.h',
    'tns.c',
    'tns.h',
    'util.c',
    'util.h',
]

libfaac = library('faac', common_src,
    include_directories: ['..', '../include'],
    c_args: c_args,
    link_args: link_args,
    dependencies: [ libm ],
    link_with: quantize_simd_libs,
    vs_module_defs: 'libfaac.def',
    install: true,
    version: '0.0.0',
    gnu_symbol_visibility: 'hidden'
)

if get_option('drm')
    drm_src = [
        'kiss_fft/kiss_fftr.c',
        'kiss_fft/kiss_fft.c',
        'kiss_fft/kiss_fft.h'
    ]
    libfaac_drm = library('faac_drm', common_src + drm_src,
        include_directories: ['..', '../include'],
        c_args: c_args + ['-DDRM'],
        link_args: link_args,
        dependencies: [ libm ],
        link_with: quantize_simd_libs,
        vs_module_defs: 'libfaac.def',
        install: true,
        version: '0.0.0',
        gnu_symbol_visibility: 'hidden'
    )
endif

pkgconfig = import('pkgconfig')
pkgconfig.generate(
    libfaac,
    name: 'FAAC',
    description: 'Freeware Advanced Audio Coder',
    version: meson.project_version(),
    filebase: 'faac'
)
