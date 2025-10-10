# This is the setup file for the AVS-saccade-locking analysis repo. It is used to install the package and its dependencies.

from setuptools import setup, find_packages

setup(name='avs_gazetime',
   version='0.1.0',
	packages=find_packages(),
	install_requires=[
		"mne",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "joblib",
        "tqdm",
        "pyyaml",
        "pyproj",
        "pyvista",
        "pyvistaqt",
        "pathlib",
        "avs_machine_room", 
    ],
	entry_points={
		'console_scripts': [
			# Add your command line scripts here
		],
	},
)

