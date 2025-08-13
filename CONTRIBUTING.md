# Contributing to `scio` ✨

Thank you for considering contributing to `scio`! It is important to the project's success and can be done in several ways.


## Code of Conduct

Everyone participating in the `scio` community, and in particular in our issue tracker, pull requests, and discussions, is expected to treat other people with respect and more generally to follow the guidelines articulated in our [Code of Conduct](CODE_OF_CONDUCT.md).


## Contribute by sharing

#### Opening issues

Open a new [issue](https://github.com/ThalesGroup/scio/issues) if you find some bug or you want to propose a new feature. Please first check that this will not create duplicates.

Make sure that your proposal is clearly written, preferably in English. In case you are reporting a bug, please include all relevant information, such as the software version and machine information. Also try to provide [Minimal Reproducible Examples](https://en.wikipedia.org/wiki/Minimal_reproducible_example).

#### Discussing the project

You can open a [discussion](https://github.com/ThalesGroup/scio/discussions) for any topic related with this package. Do you have doubts about how to use the package? Open a discussion! Do you want to show related projects, recent research or some use case for this software? Open a discussion!

You are also encouraged to answer the discussions of other users and participate actively in the discussions forum.

#### Improving the documentation

Do you feel that the documentation could be clearer? Did you find a typo? Do not hesitate to report that in an [issue](https://github.com/ThalesGroup/scio/issues) or even propose your own fix in a [pull request](https://github.com/ThalesGroup/scio/pulls). Identifying even a single misspell is relevant, there's no such thing as a "useless" issue or PR!

Advanced users can also propose the addition of new pages and examples. In case you want to do that, please open an issue to discuss that first.

#### Contributing software

You can improve this package by adding new functionality, solving pending bugs or implementing accepted feature requests. Please discuss that first to ensure that it will be accepted and to assign that to you in order to prevent duplicated efforts.

In any case, make sure that you own the rights to the software and are ok with releasing it under this project's [license](LICENSE).


## Contributing to the codebase

This is mainly done through [pull requests](https://github.com/ThalesGroup/scio/pulls).

#### How to make a pull request

##### 1. Fork the `scio` repository

Within GitHub, navigate to <https://github.com/ThalesGroup/scio> and fork the repository.

##### 2. Create a new branch

It is recommended to create pull requests from a specific (not `develop`) branch. This allows you have multiple pending PRs in parallel if required. Creating a new branch can be done directly on GitHub or locally (in which case its first push requires the `--set-upstream origin <pr-branch-name>` option).

##### 3. Clone the repository locally and relocate

```bash
# Choose where to clone the repo locally
cd ~/workspace
# Clone
git clone https://github.com/<your_username>/scio.git
# Enter the newly created folder
cd scio
# Move to PR branch (if already created)
git checkout <pr-branch-name>
```

##### 4. Implement your change

See [General coding practices](#general-coding-practices).

##### 5. Push you change to GitHub

Simply use `git push` when you want to update your remote branch, or `git push --set-upstream origin <pr-branch-name>` the first time if the branch was created locally.

##### 6. Create a pull request

From your branch on the GitHub website, you can initiate a branch comparison and pull request. Do not hesitate to mark your PR as **draft** if you wish to engage with the community while still working on your contribution.

#### General coding practices

To have a higher chance of being accepted, your contribution must:

- be compatible with the supported python versions listed in [pyproject.toml](pyproject.toml);
- not break any test;
- not decrease the code coverage;
- pass the static type checking done by `mypy`;
- comply with the `ruff` linting and formatting rules set up in [pyproject.toml](pyproject.toml).

For developers using `uv`, these can easily be checked with the following.

```bash
# From root directory
uv run pytest
uv run mypy
uvx ruff format --diff
uvx ruff check
```

If your contribution changes the documentation, you can rebuild it with the following – append `help` for more options.

```bash
# Ubuntu or MacOS
uv run make -C docs
```

```powershell
# Windows
uv run cmd /c "docs\make.bat"
```

#### First time codebase contributors

If you're looking for things to help with, browse our [issue tracker](https://github.com/ThalesGroup/scio/issues), in particular looking for [good first issues](https://github.com/ThalesGroup/scio/labels/good%20first%20issue).
