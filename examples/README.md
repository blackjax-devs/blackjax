# Examples

When you add a new sampling fuctions, please add an Exmaple Notebook to describe how the function is used. Before push your function and its corresponding notebook to `Main` branch, please convert the notebook into text file.

## Converting Notebook Files to Text Files

Please convert your `.ipynb` to `markdown` before commit your example to this repo. You can use the command below:

```python
jupytext example/your_example_notebook.ipynb --to myst
```

The script will produce the `Markdown` version of your example notebook. Please make sure you add only the `Markdown` version to your commit.

## Composing Documentation on Sphinx-Doc

We use `Sphinx` to generate document for this repo. You may also want to test whether all examples are executed correctly for documentation. We are highly encoruge you to do so on your local machine when you make change to any notebooks in `Examples`. Please follow instructions below:

1. Add your documentation to `docs/examples.rst`

2. Run the command below:

    ```shell
    make build-docs
    ```

3. Verify the generated documentation in `docs/builds`

### Execution Times

Since we store the example notebooks in `Markdown` format, but we will need outputs of in the documentations. `Sphinx` will handles all executions for you. The table below shows the execution times of example notebooks for your reference. These execution times were run on M1 machine.

|Examples                       | Execution Time (seconds) |
|-------------------------------|--------|
| GP_EllipticalSliceSampler     | 235.20 |
| HierarchicalBNN               | 84.45  |
| Introduction                  | 108.55 |
| LogisticRegression            | 7.06   |
| LogisticRegressionWithLatentGaussianSampler              | 7.10  |
| Pathfinder                    | 18.11  |
| PeriodicOrbitalMCMC           | 15.09  |
| SGMCMC                        | 316.38 |
| TemperedSMC                   | 16.09  |
| change_of_variable_hmc        | 45.23  |
| use_with_numpyro              | 14.11  |
