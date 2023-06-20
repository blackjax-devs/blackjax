# Guidelines for Contributing

Thank you for interested in contributing to Blackjax! We value the following contributions:

- Bug fixes
- Documentation
- High-level sampling algorithms from any family of algorithms: random walk,
  hamiltonian monte carlo, sequential monte carlo, variational inference,
  inference compilation, etc.
- New building blocks, e.g. new metrics for HMC, integrators, etc.

## How to contribute?

1. Run `pip install -r requirements.txt` to install all the dev
   dependencies.
2. Run `pre-commit run --all-files` and `make test` before pushing on the repo; CI should pass if
   these pass locally.

## Editing documentations

The Blackjax repository (and [sampling-book](https://github.com/blackjax-devs/sampling-book)) provides examples in the form of Markdown documents. [Jupytext](https://github.com/mwouts/jupytext) can be used by the users to convert their Jupyter notebooks to this format, or convert these documents to Jupyter notebooks. Examples are rendered in the [documentation](https://blackjax-devs.github.io/blackjax/).

### Load examples in a Jupyter notebook

To convert any example file to a Jupyter notebook you can use:

```shell
jupytext docs/examples/your_example_file.md --to notebook
```

you can then interact with the resulting notebook just like with any notebook.

### Convert my Jupyter notebook to markdown

If you implemented your example in a Jupyter notebook you can convert your `.ipynb` file to Markdown using the command below:

```shell
jupytext docs/examples/your_example_notebook.ipynb --to myst
```

Once the example file is converted to a Markdown file, you have two options for editing:

1. Edit the Markdown version as it is a regular Markdown file.
2. Edit the Notebook version, then convert it to a Markdown file once you finish editing with the command above. Jupytext can handle the change if the example has the same file name.

**Please make sure to only commit the Markdown file.**

### Composing Documentation on Sphinx-Doc

We use `Sphinx` to generate documents for this repo. We highly encourage you to check how your changes to the examples are rendered in the documentation:

1. Add your documentation to `docs/examples.rst`
2. Run the command below:

```shell
 make build-docs
```

3. Check the generated HTML documentation in `docs/_build`
