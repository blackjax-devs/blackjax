# Examples

The Blackjax repository provides examples in the form of Markdown documents. [Jupytext](https://github.com/mwouts/jupytext) can be used by the users to convert their Jupyter notebooks to this format, or convert these documents to Jupyter notebooks. Examples are rendered in the [documentation](https://blackjax-devs.github.io/blackjax/).

## For users

### Load examples in a Jupyter notebook

To convert any example file to a Jupyter notebook you can use:

```shell
jupytext examples/your_example_file.md --to notebook
```

you can then interact with the resulting notebook just like with any notebook.

## For contributors

### Convert my Jupyter notebook to markdown

If you implemented your example in a Jupyter notebook you can convert your `.ipynb` file to Markdown using the command below:

```shell
jupytext examples/your_example_notebook.ipynb --to myst
```

Once the example file is converted to a Markdown file, you have two options for editing:

1. Edit the Markdown version as it is a regular Markdown file.
2. Edit the Notebook version, then convert it to a Markdown file once you finish editing with the command above. Jupytext can handle the change if the example has the same file name.

**Please make sure to only commit the Markdown file.**

## Composing Documentation on Sphinx-Doc

We use `Sphinx` to generate documents for this repo. We highly encourage you to check how your changes to the examples are rendered in the documentation:

1. Add your documentation to `docs/examples.rst`
2. Run the command below:

```shell
 make build-docs
```

3. Check the generated HTML documentation in `docs/_build`

### Execution Times

`Sphinx` handles the conversion from Markdown to the rendered pages. The table below shows the execution times of example notebooks on a laptop to give you an idea of what to expect:

| Examples                                    | Execution Time (seconds) |
| ------------------------------------------- | ------------------------ |
| GP_EllipticalSliceSampler                   | 277.96                   |
| HierarchicalBNN                             | 187.60                   |
| Introduction                                | 19.10                    |
| LogisticRegression                          | 9.06                     |
| LogisticRegressionWithLatentGaussianSampler | 9.06                     |
| MultipleChains                              | 94.21                    |
| Pathfinder                                  | 26.13                    |
| PeriodicOrbitalMCMC                         | 27.13                    |
| RegimeSwitchingModel                        | 113.41                   |
| SGMCMC                                      | 406.82                   |
| SparseLogisticRegression                    | 248.71                   |
| TemperedSMC                                 | 23.10                    |
| change_of_variable_hmc                      | 106.32                   |
| aesara                                      | 19.07                    |
| numpyro                                     | 18.07                    |
| pymc                                        | 72.25                    |
| tfp                                         | 33.18                    |
