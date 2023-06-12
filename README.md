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
- [ ] Implement the resampling strategies for the subgrid pickup.
- [ ] Remove flak8: noqa from the files of the previous modules.
- [ ] Add environment.yaml and a setup.py to make it possible to use pip install -e (developer mode)

## Pre-commit Hooks
We use [pre-commit](https://pre-commit.com/) in order to try to mantain a clean repo.
Please install it and install the hooks for this repository using, before attemping to push
any commit:
```shell
$ pre-commit install
```
