Contribution
============

- [Code guidelines](#code-guidelines)
- [Feature request](#feature-request)
- [Development](#development)
- [Resolving bugs](#resolving-bugs)
- [Tests](#tests)

[tox.ini]: ./tox.ini

[google python style guide]: https://github.com/google/styleguide/blob/gh-pages/pyguide.md

[NumpyDoc]: https://numpydoc.readthedocs.io/en/latest/

[clean code]: https://medium.com/mindorks/how-to-write-clean-code-lessons-learnt-from-the-clean-code-robert-c-martin-9ffc7aef870c

[meaningful variable names]: https://medium.com/coding-skills/clean-code-101-meaningful-names-and-functions-bf450456d90c

[commit message template]: .gitmessage_template

## Code guidelines <a name="code-guidelines"></a>

* Try to follow PEP guidelines and [google python style guide]
* Use [NumpyDoc] documentation style.
* Try to follow the SOLID principles of [clean code]
* Use [meaningful variable names]
* Avoid inline comments - code should speak for itself :)
* For most important commits use [commit message template]

## Feature request <a name="feature-request"></a>

1. Open issue with full feature description

2. Wait for response from core team

## Development <a name="development"></a>

1. Open an issue

2. Create new branch and code there

3. Commit your work using [commit message template]

4. Be Sure that you are not behind the master, if so, rebase your branch

5. Make a pull request to master branch with meaningful description and
sign the CLA which is part of the PR checks 

6. [Run the tests](#run-tests) locally if needed; they will run automatically
on pull request

7. Wait for review

8. Apply / argue the code review comments

9. Clean up the commit history if needed

10. Push the changes again

11. Repeat steps 4-10 until convergence

Side note: **DO NOT USE** `git merge master` or `git pull master` to align your
branch with master, use `git rebase master` for it!

## Resolving bugs <a name="resolving-bugs"></a>

If you found a bug in the code, then great! Raise an issue an
[solve it yourself](#development) or
[assign it to others](#feature request) 

Please include the tracebacks and all important information in the bug
description

## Tests <a name="tests"></a>

### Add new tests

To create new test:

1. Place it in the same tree as a tested file but inside of `nucleus7/tests/`

2. Name it as `{original_file_name}_test.py` 

3. If the test is too slow, mark the test methods with `@pytest.mark.slow`

4. If the test must use GPU, mark it with `@pytest.mark.gpu`

Side note: **Do not use** `__init__.py` files inside of the tests! This will
make tox tests to test the local version and not the installed one.

### Run tests <a name="run-tests"></a>

To run the tests, type from `nucleus7` directory:

```bash
make test
```

or

```bash
tox
```

If you want to test it fast, use other make directives:

```bash
make test-local # will test it in your local configuration and without tox
make test-local-fast # same as test-local-without-pylint but excluding slow tests
```
