# Changelog
All notable changes to this project will be documented in this file.

This project uses [Semantic Versioning][sv].

## [Unreleased][new]

## [0.1.0][0.1.0] — 2020-04-24

### Added
- Add routines to load, normalize, and partition data from an H5 file into
  training and test sets.
- Parameterize specs for the data file.
- Add `train()` for training the CatClassifier.
- Add `test()` for testing and making predictions with the trained model.
- Add the CLI module for running the classifier from the command line.

### Changed
- Move CatClassifier.run() to CLI.run(). Running the package will train the
  model, displaying incremental cost and final accuracy for the default data set.

## [0.0.0][0.0.0] — 2020-04-05

### Added
- Create the project. An image classifier employing a deep neural network to
  identify pictures of cats.

---
_This file is composed with [GitHub Flavored Markdown][gfm]._

[gfm]: https://github.github.com/gfm/
[sv]: https://semver.org

[new]: https://github.com/petejh/catclass/compare/HEAD..v0.1.0
[0.1.0]: https://github.com/petejh/catclass/releases/tag/v0.1.0
[0.0.0]: https://github.com/petejh/catclass/releases/tag/v0.0.0
