cpp_args = []

if get_option('default_library') == 'shared'
    cpp_args += [ '-DPLUGGABLE' ]
endif

# transpose_layer
# transpose_src = files('transpose_layer.cpp')
# transpose_layer = static_library('transpose',
#     transpose_src,
#     dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
#     include_directories: include_directories('./'),
#     install: true,
#     install_dir: application_install_dir,
#     cpp_args: cpp_args)
# 
# transpose_dep = declare_dependency(
#     link_with: transpose_layer,
#     include_directories: include_directories('./'))
# 
# # rms_norm
# rms_norm_src = files('rms_norm.cpp')
# rms_norm_layer = static_library('rms_norm',
#     rms_norm_src,
#     dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
#     include_directories: include_directories('./'),
#     install: true,
#     install_dir: application_install_dir,
#     cpp_args: cpp_args)
# 
# rms_norm_dep = declare_dependency(
#     link_with: rms_norm_layer,
#     include_directories: include_directories('./'))
# 
# swiglu
# swiglu_src = files('swiglu.cpp')
# swiglu_layer = static_library('swiglu',
#     swiglu_src,
#     dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
#     include_directories: include_directories('./'),
#     install: true,
#     install_dir: application_install_dir,
#     cpp_args: cpp_args)
# 
# swiglu_dep = declare_dependency(
#     link_with: swiglu_layer,
#     include_directories: include_directories('./'))
# 
# # rotary_embedding
# rotary_emb_src = files('rotary_embedding.cpp')
# rotary_emb_layer = static_library('rotary_embedding',
#     rotary_emb_src,
#     dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
#     include_directories: include_directories('./'),
#     install: true,
#     install_dir: application_install_dir,
#     cpp_args: cpp_args)
# 
# rotary_emb_dep = declare_dependency(
#     link_with: rotary_emb_layer,
#     include_directories: include_directories('./'))
# 
# # custom_multi_head_attention_layer
# mha_src = files('custom_multi_head_attention_layer.cpp')
# mha_layer = static_library('custom_multi_head_attention_layer',
#     mha_src,
#     dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
#     include_directories: include_directories('./'),
#     install: true,
#     install_dir: application_install_dir,
#     cpp_args: cpp_args)
# 
# mha_dep = declare_dependency(
#     link_with: mha_layer,
#     include_directories: include_directories('./'))
# 
# Bert executable
bert_sources = [
    'main.cpp',
    transpose_src,
    rms_norm_src,
    swiglu_src,
    rotary_emb_src,
    mha_src
]

bert_dependencies = [
    app_utils_dep,
    nntrainer_dep,
    nntrainer_ccapi_dep,
    transpose_dep,
    rms_norm_dep,
    swiglu_dep,
    rotary_emb_dep,
    mha_dep
]

e = executable('nntrainer_bert',
    bert_sources,
    include_directories: include_directories('.'),
    dependencies: bert_dependencies,
    install: get_option('install-app'),
    install_dir: application_install_dir)
