from setuptools import setup
import versioneer

setup(
    name="smash",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Spatially distributed Modelling and ASsimilation for Hydrology",
    url="/",
    author="INRAE",
    packages=[
        "smash",
        "smash.core",
        "smash.core.simulation",
        "smash.solver",
        "smash.mesh",
        "smash.io",
        "smash.tools",
        "smash.dataset",
        "smash.tests",
        "smash.tests.core",
        "smash.tests.mesh",
        "smash.tests.io",
        "smash.tests.dataset",
    ],
    package_data={"smash": ["smash/solver/_solver*.so", "smash/mesh/_meshing*.so"]},
    include_package_data=True,
    install_requires=[
        "f90wrap",
        "numpy>=1.13",
        "pandas",
        "matplotlib",
        "h5py",
        "tqdm",
        "gdal",
        "scipy",
        "pyyaml",
        "SALib>=1.4.5",
        "terminaltables",
    ],
    zip_safe=False,
)
