# User ops.

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

package(
    default_visibility = [
#        "//tensorflow/core:__pkg__",
        "//visibility:public"
    ],
)
licenses(["notice"])

# One target for all user ops.
cc_library(
    name = "user_ops_op_lib",
    srcs = glob(["*.cc"]),
    copts = tf_copts(),
    linkstatic = 1,
    # visibility = ["//tensorflow/core:__pkg__"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "zero_out_ops.so",
    srcs = ["zero_out_ops.cc"],
)

tf_custom_op_library(
    name = "batch_and_pack_op.so",
    srcs = ["batch_and_pack_op.cc"],
)