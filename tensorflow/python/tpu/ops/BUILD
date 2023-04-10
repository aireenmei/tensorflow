load("//tensorflow:tensorflow.bzl", "pytype_library")  # buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

pytype_library(
    name = "ops",
    srcs = ["tpu_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":gen_zero_out_ops"
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:tpu_ops_gen",
        "//tensorflow/python/tpu:tpu_function",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_zero_out_ops",
    out = "gen_zero_out_ops.py",
    deps = [
        "//tensorflow/core/user_ops:zero_out_ops.so",
    ],
)