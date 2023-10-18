# Petal benchmark

I created a extractive QA benchmark based on this data:

```python
from askem.data import COVID_QA
COVID_QA.train_test_split(test_size=0.2, seed=2023)["test"]
```

Refer to this [repository](https://github.com/JasonLo/scrape-petal)
