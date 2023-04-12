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
    name = "array_record_ops.so",
    srcs = ["array_record_ops.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_array_record//cpp:array_record_reader",
        "@com_google_array_record//cpp:thread_pool",
        # "Remove dependency google/placer",
        "@com_googlesource_code_re2//:re2",
        "@com_google_riegeli//riegeli/bytes:fd_reader", 
    ],
)

tf_custom_op_library(
    name = "batch_and_pack_op.so",
    srcs = ["batch_and_pack_op.cc"],
    deps = [
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/kernels:fill_functor",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:macros",
    ],    
)