You can set CMake options in VS code by modifying your `settings.json` file. I had to enable CUDA builds to make the code in the main files work correctly with IntelliSense.

```
{
   "cmake.configureArgs": [
        "-DBUILD_WITH_TESTS=ON"
    ]
}
```

For Linux, nvcc should be part of your path. I do this in my bash script:

```
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
```

I could probably get away with using the symlink that the CUDA installer generates (`/usr/loca/cuda`), but I don't mind specificity.

I also have these in my bash profile. CUPTI = CUDA Profiling Tools Interface (this will help get NSight to work in VS Code):

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda
```

***

CUDA code can only be linked with libraries whose functions do not cross device boundaries. So you can't do device-code linking with host-code across a library boundary *unless* the device-code library is static. At least this was the case in 2018, according to [this StackOverflow post](https://stackoverflow.com/questions/48933195/cuda-build-shared-library). That's why the math, geometry, and buffer libraries are all marked as `STATIC` in their respective CMake files.

This is a [good example](https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt) of linking between CUDA libraries.
