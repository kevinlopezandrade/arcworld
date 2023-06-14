# WIP: ARCWORLD
Refactoring of the Task Generation modules implemented by Yassine and Michael
for the Arc Challange. It is still a protype, so expect changes in the API.
You can find an small example of the API in **example.py**.

## TODO
- [ ] Get rid of the deprecated module.
- [ ] Make it compliant with the static type checker Pyright or Mypy and add it as a pre-commit hook.
- [ ] Split the dsl/functional.py into two modules.
- [ ] Define a common strategy for Task Generation, after the Grid is sampled.
- [ ] Document all the modules.
- [ ] For the modules that use the DSL implement the mixers and implement the code generation module.
- [x] Implement the resampling strategies for the subgrid pickup.
- [ ] Remove flak8: noqa from the files of the previous modules.
- [x] Add environment.yaml and a setup.py to make it possible to use pip install -e (developer mode)

## Development
We recommend using [conda](https://docs.conda.io/en/latest/) as a virtual
environment manager to avoid conflicts with the hosting operating system.
```shell
$ conda create -n arcworld python=3.9
```

We use [poetry](https://python-poetry.org/) as our dependency manager and build
system framework. Please install it using:
```shell
$ pip install poetry
```
and install the dependencies by running:
```shell
$ poetry update
```

Install the library in editable mode so that changes in the codebase
can be tested easily.

```shell
$ pip install -e .
```

## Contributing
### Pre-Commit Hooks
We use [pre-commit](https://pre-commit.com/) in order to try to mantain a clean repo.
Please install it using:
```shell
$ pip instal pre-commit
```
and install the hooks before attemping to push any commit:
```shell
$ pre-commit install
```
