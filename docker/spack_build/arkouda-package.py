from spack_repo.builtin.build_systems.makefile import MakefilePackage
from spack.package import *

class Arkouda(MakefilePackage):
    homepage = "https://github.com/Bears-R-Us/arkouda"
    url      = "https://github.com/Bears-R-Us/arkouda/archive/refs/tags/v2024.10.02.tar.gz"
    git      = "https://github.com/Bears-R-Us/arkouda.git"
    license("MIT")
    maintainers("ajpotts", "arezaii")

    version("main", branch="main")
    version("2025.08.20", sha256="3e305930905397ff3a7a28a5d8cc2c9adca4194ca7f6ee51f749f427a2dea92c")
    version("2025.07.03", sha256="eb888fac7b0eec6b4f3bfa0bfe14e5c8f15b449286e84c45ba95c44d8cd3917a")
    version("2025.01.13", sha256="bb53bab92fedf43a47aadd9195eeedebe5f806d85887fa508fb5c69f2a4544ea")
    version("2024.12.06", sha256="92ca11319a9fdeeb8879afbd1e0c9c1b1d14aa2496781c1481598963d3c37b46")
    version("2024.10.02", sha256="00671a89a08be57ff90a94052f69bfc6fe793f7b50cf9195dd7ee794d6d13f23")
    version("2024.06.21", sha256="ab7f753befb3a0b8e27a3d28f3c83332d2c6ae49678877a7456f0fcfe42df51c")

    variant("distributed", default=False,
            description="Build for multi-locale execution")
            
    # inside class Arkouda(MakefilePackage):

    # Arrow with all the codecs you enable
    depends_on("arrow +parquet +snappy +zlib +lz4 +brotli +bz2 +zstd",
               type=("build", "link", "run", "test"))
    variant("lz4", default=True,
            description="Enable Parquet LZ4 support via Arrow")


    # Ensure Arrow sees a CMake-built LZ4 (fixes the lz4Alt/LZ4_LIB cmake lookup)
    depends_on("lz4 build_system=cmake", type=("build", "link"))

    # Help Arrow’s find-modules by ensuring pkg-config is around when we want lz4
    depends_on("pkgconf", when="+lz4", type="build")

    # (…keep your Chapel/libzmq/hdf5/etc. deps as they are…)


    # Chapel ranges (keep your split across Arkouda versions)
    depends_on("chapel@2.0.0:2.4.99 +hdf5 +zmq", type=("build","link","run","test"))



    depends_on("cmake@3.13.4:", type="build")
    depends_on("python@3.9:", type=("build","link","run","test"))
    depends_on("libzmq@4.2.5:", type=("build","link","run","test"))
    depends_on("hdf5+hl~mpi", type=("build","link","run","test"))
    depends_on("libiconv", type=("build","link","run","test"))
    depends_on("libidn2", type=("build","link","run","test"))




    requires("^chapel comm=none", when="~distributed")
    requires("^chapel +python-bindings", when="@2024.10.02:")
    requires("^chapel comm=gasnet", "^chapel comm=ugni", "^chapel comm=ofi",
             policy="one_of", when="+distributed")

    patch("makefile-fpic-2024.06.21.patch", when="@2024.06.21")
    patch("makefile-fpic-2024.10.02.patch", when="@2024.10.02:")

    sanity_check_is_file = [join_path("bin", "arkouda_server")]

    def check(self):
        pass  # tests need the Python client

    def edit(self, spec, prefix):
        self.update_makefile_paths(spec, prefix)

    def update_makefile_paths(self, spec, prefix):
        with open("Makefile.paths", "w") as f:
            f.write("$(eval $(call add-path,{0}))\n".format(spec["hdf5"].prefix))
            f.write("$(eval $(call add-path,{0}))\n".format(spec["libzmq"].prefix))
            f.write("$(eval $(call add-path,{0}))\n".format(spec["arrow"].prefix))
            f.write("$(eval $(call add-path,{0}))\n".format(spec["libiconv"].prefix))
            f.write("$(eval $(call add-path,{0}))\n".format(spec["libidn2"].prefix))
            if spec.satisfies("+lz4"):
                f.write("$(eval $(call add-path,{0}))\n".format(spec["lz4"].prefix))


    def build(self, spec, prefix):
        if spec.satisfies("+distributed"):
            with set_env(ARKOUDA_SKIP_CHECK_DEPS="1"):
                tty.warn("Distributed build detected. Skipping dependency checks")
                make()
        else:
            make()

    def install(self, spec, prefix):
        mkdir(prefix.bin)
        install("arkouda_server", prefix.bin)
        if spec.satisfies("+distributed"):
            install("arkouda_server_real", prefix.bin)

