# Tensorboard Plugins for NNP Profiling

Follow these steps to build the demo plugins for NNP compatibility and memory usage in Docker. 

#### Clone this branch and navigate to the project directory
```
git clone https://github.com/mhbuehler/tensorboard.git --branch melanie/nnp_plugins_1.14
cd tensorboard
```

#### Build a Docker container for Bazel 0.24.1 
Using the Dockerfile provided, build an image with the Tensorboard 1.14 build environment:
```
docker build \
    -f tensorboard/plugins/Dockerfile \
    -t bazel:0.24.1 \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} .
```

#### Run a container with mounted volumes for the project at `/tensorboard` and the sample data at `/tmp`
```
docker run \
    -it \
    -v $PWD:/tensorboard \
    -v $PWD/tensorboard/plugins/sample_data:/tmp \
    -p 6006:6006 \
    --name nnp_plugins \
    bazel:0.24.1
```

#### Build the histogram demo and run Tensorboard in the container
```
cd tensorboard
bazel run tensorboard/plugins/histogram:histograms_demo
bazel run tensorboard -- --logdir=/tmp
```

#### Go to `http://localhost:6006/` in your browser
You will see the "Graphs" view and compatibility checker first. 
Click the "NNP Compatibility" radio button and then the graph nodes to see more information.
![Compatibility Checker](compatibility-checker.png)

Click the "Histograms" tab to see the memory usage plugin.
![Memory Usage](memory-usage.png)

#### Build a wheel file to test the plugins in a container with nprof
From inside the `/tensorboard` directory:
```
bazel run //tensorboard/pip_package:build_pip_package
```
The wheel file at `/tmp/tensorboard/dist/tensorboard-1.14.0-py3-none-any.whl` will be available outside the container in the `sample_data` directory, since it was mounted to `/tmp`.