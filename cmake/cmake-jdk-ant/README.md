# cmake-jdk-ant

Contact: Balthasar Reuter (balthasar.reuter@ecmwf.int)

A CMake project to bootstrap OpenJDK and Ant during the configuration phase.

Variables influencing the behaviour of this CMake configuration:

* `MINIMUM_JAVA_VERSION`: The minimum JDK and JRE version that should be available. If no Java is found or a version too old, OpenJDK will be bootstrapped. Default: `11`
* `MINIMUM_ANT_VERSION`: The minimum Ant version that should be available. If no Ant is found or a version too old, Ant will be bootstrapped. Default: `1.10`
* `FORCE_OPEN_JDK_INSTALL`: Force bootstrapping of OpenJDK, regardless of any available version. Default: `OFF`
* `FORCE_ANT_INSTALL`: Force bootstrapping of Ant, regardless of any available version. Default: `OFF`
* `OPEN_JDK_INSTALL_VERSION`: The OpenJDK version to install. Default: `11.0.2`
* `ANT_INSTALL_VERSION`: The Ant version to install. Default: `1.10.12`
* `OPEN_JDK_MIRROR`: Allows to set an alternative mirror for OpenJDK download.
* `ANT_MIRROR`: Allows to set an alternative mirror for Ant download.

The purpose of this is to provide a way of on-the-fly installation of Java/Ant toolchain dependencies on systems where no usable setup is available.

## Example:

In a project that requires Java and Ant, add `cmake-jdk-ant` as a subdirectory:

```cmake
...
add_subdirectory( cmake-jdk-ant )
...
```

Subsequently, any calls to `find_package( Java )` will yield the bootstrapped toolchain.
