# pyActigraphy

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pipeline status](https://gitlab.uliege.be/CyclotronResearchCentre/Studies/CogNap/Actigraphy/pyActigraphy/badges/master/pipeline.svg)](https://gitlab.uliege.be/CyclotronResearchCentre/Studies/CogNap/Actigraphy/pyActigraphy/commits/master)

Analyse package for actigraphy data


## Release strategy

The package is released (See [Releases](https://github.com/CyclotronResearchCentre/pyActigraphy/releases)) on a regular basis in a two step fashion:
1. vX.Y.Z-beta versions are made available for testing purposes.
2. vX.Y.Z versions are made available for public use.

**WARNING** Do not use un-released or beta version of the package for analysis.

Please, make sure you use the [latest](https://github.com/CyclotronResearchCentre/pyActigraphy/releases/latest) release before creating an issue or suggesting an improvement.

## Prerequisites

- python 3.X

## Local installation

In a (bash) shell, to use the *vX.Y* version of the pyActigraphy package, simply type:

```bash
git clone git@github.com:CyclotronResearchCentre/pyActigraphy.git
cd pyActigraphy/
git fetch --tags
git checkout vX.Y
pip install -e .
```

## Tutorials

[pyActigraphy-Tutorial](https://github.com/CyclotronResearchCentre/pyActigraphy-Tutorial): Slides presenting the overall project as well as notebooks illustrating how to use the pyActigraphy package.

## Contributing

There are plenty of ways to contribute to this package, including (but not limiting to):
- report bugs (and, ideally, how to reproduce the bug)
- suggest improvements
- improve the documentation
- hug or high-five the authors when you meet them!

## Authors

* **Grégory Hammad** [@ghammad](https://github.com/ghammad) - *Initial and main developer*
* **Mathilde Reyt** [@ReytMathilde](https://github.com/ReytMathilde)

See also the list of [contributors](https://github.com/CyclotronResearchCentre/pyActigraphy/contributors) who participated in this project.

## License

This project is licensed under the GNU GPL-3.0 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* **Aubin Ardois** [@aardoi](https://github.com/aardoi) developed the first version of the MTN class during his internship at the CRC, in May-August 2018.
* The CRC colleagues for their support, ideas, etc.

<!-- [license-badge]: https://img.shields.io/npm/l/all-contributors.svg?style=flat-square -->
