# Contributing

Thanks for considering contributing! Please read this document to learn the various ways you can contribute to this project and how to go about doing it.

## Bug reports and feature requests

### Did you find a bug?

First, do [a quick search](https://github.com/py-why/pywhy-graphs/issues) to see whether your issue has already been reported.
If your issue has already been reported, please comment on the existing issue.

Otherwise, open [a new GitHub issue](https://github.com/py-why/pywhy-graphs/issues).  Be sure to include a clear title
and description.  The description should include as much relevant information as possible.  The description should
explain how to reproduce the erroneous behavior as well as the behavior you expect to see.  Ideally you would include a
code sample or an executable test case demonstrating the expected behavior.

### Do you have a suggestion for an enhancement or new feature?

We use GitHub issues to track feature requests. Before you create an feature request:

* Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing
it first on a GitHub issue.
* Check the documentation to make sure your feature does not already exist.
* Do [a quick search](https://github.com/py-why/pywhy-graphs/issues) to see whether your feature has already been suggested.

When creating your request, please:

* Provide a clear title and description.
* Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.
* Include code examples to demonstrate how the enhancement would be used.

## Making a pull request

When you're ready to contribute code to address an open issue, please follow these guidelines to help us be able to review your pull request (PR) quickly.

1. **Initial setup** (only do this once)

    <details><summary>Expand details 👇</summary><br/>

    If you haven't already done so, please [fork](https://help.github.com/en/enterprise/2.13/user/articles/fork-a-repo) this repository on GitHub.

    Then clone your fork locally with

        git clone https://github.com/USERNAME/pywhy-graphs.git

    or 

        git clone git@github.com:USERNAME/pywhy-graphs.git

    At this point the local clone of your fork only knows that it came from *your* repo, github.com/USERNAME/pywhy-graphs.git, but doesn't know anything the *main* repo, [https://github.com/py-why/pywhy-graphs.git](https://github.com/py-why/pywhy-graphs). You can see this by running

        # Note you should be in the "pywhy-graphs" directory. If you're not
        # run "cd ./pywhy-graphs" to change directory into the repo
        git remote -v

    which will output something like this:

        origin https://github.com/USERNAME/pywhy-graphs.git (fetch)
        origin https://github.com/USERNAME/pywhy-graphs.git (push)

    This means that your local clone can only track changes from your fork, but not from the main repo, and so you won't be able to keep your fork up-to-date with the main repo over time. Therefore you'll need to add another "remote" to your clone that points to [https://github.com/py-why/pywhy-graphs.git](https://github.com/py-why/pywhy-graphs). To do this, run the following:

        git remote add upstream https://github.com/py-why/pywhy-graphs.git

    Now if you do `git remote -v` again, you'll see

        origin https://github.com/USERNAME/pywhy-graphs.git (fetch)
        origin https://github.com/USERNAME/pywhy-graphs.git (push)
        upstream https://github.com/py-why/pywhy-graphs.git (fetch)
        upstream https://github.com/py-why/pywhy-graphs.git (push)

    Finally, you'll need to create a Python 3 virtual environment suitable for working on this project. There a number of tools out there that making working with virtual environments easier.
    The most direct way is with the [`venv` module](https://docs.python.org/3.7/library/venv.html) in the standard library, but if you're new to Python or you don't already have a recent Python 3 version installed on your machine,
    we recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

    On Mac, for example, you can install Miniconda with [Homebrew](https://brew.sh/):

        brew install miniconda

    Then you can create and activate a new Python environment by running:

        conda create -n pywhy-graphs python=3.9
        conda activate pywhy-graphs

    Once your virtual environment is activated, you can install your local clone in "editable mode" with

        pip install -U pip setuptools wheel
        pip install -e .[dev]

    The "editable mode" comes from the `-e` argument to `pip`, and essential just creates a symbolic link from the site-packages directory of your virtual environment to the source code in your local clone. That way any changes you make will be immediately reflected in your virtual environment.

    </details>

2. **Ensure your fork is up-to-date**

    <details><summary>Expand details 👇</summary><br/>

    Once you've added an "upstream" remote pointing to [https://github.com/allenai/python-package-temlate.git](https://github.com/py-why/pywhy-graphs), keeping your fork up-to-date is easy:

        git checkout main  # if not already on main
        git pull --rebase upstream main
        git push

    </details>

3. **Create a new branch to work on your fix or enhancement**

    <details><summary>Expand details 👇</summary><br/>

    Committing directly to the main branch of your fork is not recommended. It will be easier to keep your fork clean if you work on a separate branch for each contribution you intend to make.

    You can create a new branch with

        # replace BRANCH with whatever name you want to give it
        git checkout -b BRANCH
        git push -u origin BRANCH

    </details>

4. **Developing and testing your changes**

    <details><summary>Expand details 👇</summary><br/>

    Our continuous integration (CI) testing runs [a number of checks](https://github.com/py-why/pywhy-graphs/actions) for each pull request on [GitHub Actions](https://github.com/features/actions). You can run most of these tests locally, which is something you should do *before* opening a PR to help speed up the review process and make it easier for us. Please see our [development guide](https://github.com/py-why/pywhy-graphs/blob/main/DEVELOPING.md) for a comprehensive overview of useful commands leveraging [poetry](https://python-poetry.org). This will cover aspects of code style checking, unit testing, integration testing, and building the documentation. We try to make it as easy as possible with copy/paste commands leveraging poetry which will guide your development process!

    And finally, please update the [CHANGELOG](https://github.com/py-why/pywhy-graphs/docs/whats_new.rst) with notes on your contribution in the "Unreleased" section at the top.

    After all of the above checks have passed, you can now open [a new GitHub pull request](https://github.com/py-why/pywhy-graphs/pulls).
    Make sure you have a clear description of the problem and the solution, and include a link to relevant issues.

    We look forward to reviewing your PR!

    </details>

### Writing docstrings

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to build our API docs, which automatically parses all docstrings
of public classes and methods. All docstrings should adhere to the [Numpy styling convention](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html).

### Testing Changes Locally With Poetry
With poetry installed, we have included a few convenience functions to check your code. **These checks must pass** and will be checked by the PR's continuous integration services. You can install the various different developer dependencies with poetry:

    poetry install --with style, docs, test

You can verify that your code will pass certain style, formatting and lint checks by running:

    poetry run poe verify

``verify`` runs a sequence of tests that can also be run individually. For example, you can check code formatting with black:

    poetry run poe format_check

If you would like to automatically black format your changes:

    poetry run poe format

You can then check for code style and general linting:

    poetry run poe lint

Finally, you should run some mypy type checks:

    poetry run poe type_check

### Pre-commit Hooks

To improve development workflows, you can use pre-commit hooks that automatically check for all of the `poe` tasks above:
    
    pre-commit install

Whenever a commit is performed, these hooks will run to ensure that the code satisfies all formatting, style, and test checks.

### Documentation

If you need to build the documentation locally and check for doc errors:

    poetry run poe build_docs

### Dependency Changes

If you need to add new, or remove old dependencies, then you need to modify the ``pyproject.toml`` file and then also update the ``poetry.lock`` file, which version-controls all necessary dependencies. If you alter any dependency in the ``pyproject.toml`` file, you must run:

    poetry update

To update the lock file.
