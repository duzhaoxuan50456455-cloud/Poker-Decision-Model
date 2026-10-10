# Tests

Only a package import smoke test exists. Future tests should cover data contracts,
group split leakage, dynamic option masking/permutation, gradient flow, metrics,
and checkpoint round trips. Keep unit tests offline; separate model downloads
and live API integration checks.
