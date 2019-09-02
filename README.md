| [Linux][lin-link] |  [Codecov][cov-link]  |
| :---------------: | :-------------------: |
| ![lin-badge]      | ![cov-badge]          |

[lin-badge]: https://github.com/phillyfan1138/cf_dist_utils_rust/workflows/Rust/badge.svg
[lin-link]:  https://github.com/phillyfan1138/cf_dist_utils_rust/actions
[cov-badge]: https://codecov.io/gh/phillyfan1138/cf_dist_utils_rust/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/cf_dist_utils_rust

# cf_dist_utils

This is a set of utilities to wrap around my [fang_oost](https://github.com/phillyfan1138/fang_oost_rust) library. Documentation is on [docs.rs](https://docs.rs/cf_dist_utils/0.6.1/cf_dist_utils/). The main features are the ability to compute VaR, the CDF of a distribution, and the expected shortfall of a distribution.

## using cf_dist_utils
Put the following in your Cargo.toml:

```toml
[dependencies]
cf_dist_utils = "0.8"
```

Import and use:

```rust
extern crate num_complex;
use num_complex::Complex;
extern crate cf_dist_utils;
let mu=2.0;
let sigma=5.0;
let num_u=128;
let x_min=-20.0;
let x_max=25.0;
let max_iterations=1000;
let tolerance=0.0000001;
let alpha=0.05;
let norm_cf=|u:&Complex<f64>| (u*mu+0.5*sigma*sigma*u*u).exp();
let (estimated_es, estimated_var)=cf_dist_utils::get_expected_shortfall_and_value_at_risk(
    alpha, num_u, x_min, x_max, max_iterations, tolerance, norm_cf
);
```

# Benchmarks

https://phillyfan1138.github.io/cf_dist_utils_rust/report/index.html