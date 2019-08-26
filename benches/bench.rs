#[macro_use]
extern crate criterion;
extern crate cf_dist_utils;
extern crate fang_oost;
extern crate num_complex;
use num_complex::Complex;
use criterion::Criterion;
fn bench_variance(c: &mut Criterion) {
    let mu = 2.0;
    let sigma = 5.0;
    let num_u = 128;
    let x_min = -40.0;
    let x_max = 45.0;
    let norm_cf = |u: &Complex<f64>| (u * mu + 0.5 * sigma * sigma * u * u).exp();
    let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, norm_cf);
    c.bench_function("gaussian variance", move |b|{
        b.iter(|| cf_dist_utils::get_variance_discrete_cf(x_min, x_max, &discrete_cf));
    });
}

fn bench_value_at_risk(c: &mut Criterion) {
    let mu = 2.0;
    let sigma = 5.0;
    let num_u = 128;
    let x_min = -20.0;
    let x_max = 25.0;
    let alpha = 0.05;
    let norm_cf = move |u: &Complex<f64>| (u * mu + 0.5 * sigma * sigma * u * u).exp();
    c.bench_function("gaussian value at risk", move |b|{
        b.iter(|| cf_dist_utils::get_expected_shortfall_and_value_at_risk(
            alpha, num_u, x_min, x_max, 100, 0.0000001, &norm_cf,
        ));
    });
}
criterion_group!(benches, bench_variance, bench_value_at_risk);
criterion_main!(benches);
