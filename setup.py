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
        "smash.core.model",
        "smash.core.signal_analysis",
        "smash.core.signal_analysis.segmentation",
        "smash.core.signal_analysis.signatures",
        "smash.core.signal_analysis.scores",
        "smash.core.simulation",
        "smash.core.simulation.run",
        "smash.core.simulation.optimize",
        "smash.factory",
        "smash.factory.dataset",
        "smash.factory.mesh",
        "smash.factory.net",
        "smash.factory.samples",
        "smash.fcore",
        "smash.io",
        "smash.io.mesh",
        "smash.io.setup",
        "smash.tests",
        "smash.tests.core",
        "smash.tests.core.signal_analysis",
        "smash.tests.core.simulation",
        "smash.tests.factory",
        "smash.tests.io",
    ],
    package_data={"smash": ["smash/fcore/*.so", "smash/factory/mesh/*.so"]},
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
        "terminaltables",
    ],
    zip_safe=False,
)
