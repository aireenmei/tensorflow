# Description:
#   Contains User Ops (internal TensorFlow version).

load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

visibility = [
    "//tensorflow:internal",
]

package(
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_gen_op_wrapper_private_py(
    name = "user_ops_gen",
)

# This target is deprecated.
py_library(
    name = "ops",
    srcs = ["user_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":user_ops_gen",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_zero_out_ops",
    out = "gen_zero_out_ops.py",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/user_ops:zero_out_ops.so",
    ],
)

py_library(
    name = "zero_out_ops_py",
    srcs = ["zero_out_ops.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":gen_zero_out_ops",
    ],
)