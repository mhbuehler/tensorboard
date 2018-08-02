package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_custom_dashboard",
    srcs = [
        "tf-custom-dashboard.html",
        "tf-custom-loader.html",
    ],
    path = "/tf-custom-dashboard",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_card_heading",
        "//tensorboard/components/tf_categorization_utils",
        "//tensorboard/components/tf_color_scale",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:lodash",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_paginated_view",
        "//tensorboard/components/tf_runs_selector",
        "//tensorboard/components/tf_tensorboard:registry",
        "@org_polymer_iron_icon",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
    ],
)

tf_web_library(
    name = "demo",
    srcs = ["index.html"],
    path = "/tf-custom-dashboard",
    deps = [
        ":tf_custom_dashboard",
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_imports:webcomponentsjs",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
    ],
)
