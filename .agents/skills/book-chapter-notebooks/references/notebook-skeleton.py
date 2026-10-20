# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # {{Chapter title}}: implementation
#
# {{One sentence: what the notebook implements and checks.}}
# Companion to the chapter note: [{{note title}}](../chapters/{{note file name}}.md).
#
# **Sections**
# 1. {{Section one}}
# 2. {{Section two}}
# 3. Checkpoints

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.figsize": (7, 4), "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False})

SEED = 0  # per-section seeds below derive from named constants, never from a shared generator
{{CONSTANT}} = {{value}}

# %% [markdown]
# ## 1. {{Section title}}
#
# {{What it is and why, in plain words.}}
#
# $$
# {{update rule in the book's notation}}
# $$
#
# - {{what it estimates / what each symbol is}}

# %%
def {{method}}(...) -> ...:
    """{{What it estimates, in one line.}}"""
    ...


# %%
rng = np.random.default_rng(SEED + 1)
# ... seeded experiment ...
print("Checkpoint: {{what was verified}}")

# %% [markdown]
# ### Reading the result
#
# - {{What the output shows; every number equals the printed output.}}

# %% [markdown]
# ## Checkpoints

# %%
# assert statements against known truth or tolerances, then a short result table
