from setuptools import find_packages, setup


setup(
    name="sims-stylized-hsi",
    version="0.1.0",
    description="SIMS motion encoder, text-conditioned policies and FSM inference",
    author="Wenjia Wang",
    packages=find_packages(),
    python_requires=">=3.8,<3.9",
    include_package_data=True,
    license_files=["third_party/ASE_LICENSE.txt"],
    package_data={"sims": ["data/cfg/*.yaml", "data/cfg/train/rlg/*.yaml", "data/assets/mjcf/*.xml", "data/assets/mjcf/*.urdf"]},
)
