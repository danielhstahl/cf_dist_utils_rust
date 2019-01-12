//! Contains functions for computing the partial expectation, quantile, and cumulative density
//! function given a characteristic function.  
extern crate fang_oost;
extern crate roots;
use roots::find_root_regula_falsi;
use roots::SimpleConvergency;
use std::error::Error;
use std::fmt;
extern crate num_complex;
extern crate rayon;
#[macro_use]
#[cfg(test)]
extern crate approx;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
use num_complex::Complex;

use rayon::prelude::*;

#[derive(Debug)]
pub enum ValueAtRiskError {
    Alpha,
    Root(String),
}

impl fmt::Display for ValueAtRiskError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ValueAtRiskError::Alpha => write!(f, "Alpha needs to be within 0 and 1!"),
            ValueAtRiskError::Root(desc) => write!(f, "{}", desc),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RiskMetric {
    pub expected_shortfall: f64,
    pub value_at_risk: f64,
}

/**
    Function to compute the CDF of a distribution; see
    http://danielhstahl.com/static/media/CreditRiskExtensions.c31991d2.pdf

*/
fn vk_cdf(u: f64, x: f64, a: f64, k: usize) -> f64 {
    if k == 0 {
        x - a
    } else {
        ((x - a) * u).sin() / u
    }
}

fn diff_pow(x: f64, a: f64) -> f64 {
    0.5 * (x.powi(2) - a.powi(2))
}

/**
    Function to compute the partial expectation of a distribution; see
    http://danielhstahl.com/static/media/CreditRiskExtensions.c31991d2.pdf.
*/
fn vk_pe(u: f64, x: f64, a: f64, k: usize) -> f64 {
    let arg = (x - a) * u;
    let u_den = 1.0 / u;
    if k == 0 {
        diff_pow(x, a)
    } else {
        x * arg.sin() * u_den + u_den.powi(2) * (arg.cos() - 1.0)
    }
}
/**
    Function to compute the partial expectation squared of a distribution.
*/
fn vk_pv(u: f64, x: f64, c: f64, a: f64, k: usize) -> f64 {
    let arg = (x - a) * u;
    let u_den = 1.0 / u;
    if k == 0 {
        -(a - x).powi(3) / 3.0 - (a - x).powi(2) * (x - c) - (a - x) * (x - c).powi(2)
    } else {
        (x - c).powi(2) * arg.sin() * u_den - 2.0 * arg.sin() * u_den.powi(3)
            + 2.0 * ((x - c) * arg.cos() + c - a) * u_den.powi(2)
    }
}

fn compute_value_at_risk(
    alpha: f64,
    x_min: f64,
    x_max: f64,
    max_iterations: usize,
    tolerance: f64,
    discrete_cf: &[Complex<f64>],
) -> Result<f64, ValueAtRiskError> {
    let vf = |u, x, u_index| vk_cdf(u, x, x_min, u_index);
    let in_f = |x: f64| {
        fang_oost::get_expectation_single_element_real(x_min, x_max, x, discrete_cf, vf) - alpha
    };
    let mut convergency = SimpleConvergency {
        eps: tolerance,
        max_iter: max_iterations,
    };
    match find_root_regula_falsi(x_min, x_max, &in_f, &mut convergency) {
        Ok(v) => Ok(-v),
        Err(e) => Err(ValueAtRiskError::Root(e.description().to_string())),
    }
}
fn compute_expected_shortfall(
    alpha: f64,
    x_min: f64,
    x_max: f64,
    value_at_risk: f64,
    discrete_cf: &[Complex<f64>],
) -> f64 {
    -fang_oost::get_expectation_single_element_real(
        x_min,
        x_max,
        -value_at_risk,
        discrete_cf,
        |u, x, u_index| vk_pe(u, x, x_min, u_index),
    ) / alpha
}
/// Returns expected shortfall (partial expectation) and value at risk (quantile)
/// given a discrete characteristic function.
///
/// # Remarks
/// Technically there is no guarantee of convergence for value at risk.
/// The cosine expansion oscillates and the value at risk may be under
/// or over stated.  However, in tests it appears to converge for a wide
/// range of distributions
///
/// # Examples
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// #[macro_use]
/// extern crate approx;
/// extern crate cf_dist_utils;
/// # fn main(){
/// let mu=2.0;
/// let sigma=5.0;
/// let num_u=128;
/// let x_min=-20.0;
/// let x_max=25.0;
/// let alpha=0.05;
/// let max_iterations=1000;
/// let tolerance=0.0001;
/// let norm_cf=vec![Complex::new(1.0, 1.0), Complex::new(-1.0, 1.0)];
/// let cf_dist_utils::RiskMetric{
///     expected_shortfall,
///     value_at_risk
/// }=cf_dist_utils::get_expected_shortfall_and_value_at_risk_discrete_cf(
///     alpha, x_min, x_max, max_iterations, tolerance, &norm_cf
/// ).unwrap();
/// # }
/// ```
pub fn get_expected_shortfall_and_value_at_risk_discrete_cf(
    alpha: f64,
    x_min: f64,
    x_max: f64,
    max_iterations: usize,
    tolerance: f64,
    discrete_cf: &[Complex<f64>],
) -> Result<RiskMetric, ValueAtRiskError> {
    if alpha > 0.0 && alpha < 1.0 {
        let value_at_risk =
            compute_value_at_risk(alpha, x_min, x_max, max_iterations, tolerance, &discrete_cf)?;
        let expected_shortfall =
            compute_expected_shortfall(alpha, x_min, x_max, value_at_risk, &discrete_cf);
        Ok(RiskMetric {
            expected_shortfall,
            value_at_risk,
        })
    } else {
        Err(ValueAtRiskError::Alpha)
    }
}
/// Returns expectation
/// given a discrete characteristic function.
///
/// # Examples
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// #[macro_use]
/// extern crate approx;
/// extern crate cf_dist_utils;
/// # fn main(){
/// let mu=2.0;
/// let sigma=5.0;
/// let num_u=128;
/// let x_min=-20.0;
/// let x_max=25.0;
/// let norm_cf=vec![Complex::new(1.0, 1.0), Complex::new(-1.0, 1.0)];
/// let expectation=cf_dist_utils::get_expectation_discrete_cf(
///     x_min, x_max, &norm_cf
/// );
/// # }
/// ```
pub fn get_expectation_discrete_cf(x_min: f64, x_max: f64, discrete_cf: &[Complex<f64>]) -> f64 {
    fang_oost::get_expectation_single_element_real(
        x_min,
        x_max,
        x_max,
        discrete_cf,
        |u, x, u_index| vk_pe(u, x, x_min, u_index),
    )
}
/// Returns variance
/// given a discrete characteristic function.
///
/// # Examples
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// #[macro_use]
/// extern crate approx;
/// extern crate cf_dist_utils;
/// # fn main(){
/// let mu=2.0;
/// let sigma=5.0;
/// let num_u=128;
/// let x_min=-20.0;
/// let x_max=25.0;
/// let norm_cf=vec![Complex::new(1.0, 1.0), Complex::new(-1.0, 1.0)];
/// let variance=cf_dist_utils::get_variance_discrete_cf(
///     x_min, x_max, &norm_cf
/// );
/// # }
/// ```
pub fn get_variance_discrete_cf(x_min: f64, x_max: f64, discrete_cf: &[Complex<f64>]) -> f64 {
    let expectation = fang_oost::get_expectation_single_element_real(
        x_min,
        x_max,
        x_max,
        discrete_cf,
        |u, x, u_index| vk_pe(u, x, x_min, u_index),
    );
    fang_oost::get_expectation_single_element_real(
        x_min,
        x_max,
        x_max,
        discrete_cf,
        |u, x, u_index| vk_pv(u, x, expectation, x_min, u_index),
    )
}

/// Returns expected shortfall (partial expectation) and value at risk (quantile)
/// given a characteristic function.
///
/// # Remarks
/// Technically there is no guarantee of convergence for value at risk.
/// The cosine expansion oscillates and the value at risk may be under
/// or over stated.  However, in tests it appears to converge for a wide
/// range of distributions
///
/// # Examples
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// #[macro_use]
/// extern crate approx;
/// extern crate cf_dist_utils;
/// # fn main(){
/// let mu=2.0;
/// let sigma=5.0;
/// let num_u=128;
/// let x_min=-20.0;
/// let x_max=25.0;
/// let max_iterations=1000;
/// let tolerance=0.0000001;
/// let alpha=0.05;
/// let norm_cf=|u:&Complex<f64>| (u*mu+0.5*sigma*sigma*u*u).exp();
/// let reference_var=6.224268;
/// let reference_es=8.313564;
/// let cf_dist_utils::RiskMetric{
///     expected_shortfall,
///     value_at_risk
/// }=cf_dist_utils::get_expected_shortfall_and_value_at_risk(
///     alpha, num_u, x_min, x_max, max_iterations, tolerance, norm_cf
/// ).unwrap();
/// assert_abs_diff_eq!(reference_var, value_at_risk, epsilon=0.0001);
/// assert_abs_diff_eq!(reference_es, expected_shortfall, epsilon=0.001);
/// # }
/// ```
pub fn get_expected_shortfall_and_value_at_risk<T>(
    alpha: f64,
    num_u: usize,
    x_min: f64,
    x_max: f64,
    max_iterations: usize,
    tolerance: f64,
    fn_inv: T,
) -> Result<RiskMetric, ValueAtRiskError>
where
    T: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, fn_inv);
    get_expected_shortfall_and_value_at_risk_discrete_cf(
        alpha,
        x_min,
        x_max,
        max_iterations,
        tolerance,
        &discrete_cf,
    )
}

/// Returns vector of cumulative density function given a characteristic function.
///  
/// # Examples
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_dist_utils;
/// # fn main(){
/// let mu = 2.0;
/// let sigma = 5.0;
/// let num_u = 128;
/// let num_x = 1024;
/// let x_min=-20.0;
/// let x_max=25.0;
/// let norm_cf=|u:&Complex<f64>| (u*mu+0.5*sigma*sigma*u*u).exp();
/// let cdf=cf_dist_utils::get_cdf(
///     num_x, num_u, x_min, x_max, &norm_cf
/// );
/// # }
/// ```
pub fn get_cdf<T>(num_x: usize, num_u: usize, x_min: f64, x_max: f64, cf: T) -> Vec<f64>
where
    T: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let x_domain = fang_oost::get_x_domain(num_x, x_min, x_max);
    let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, &cf);
    let result =
        fang_oost::get_expectation_real(x_min, x_max, x_domain, &discrete_cf, |u, x, u_index| {
            vk_cdf(u, x, x_min, u_index)
        })
        .collect();
    result
}

/// Returns cumulative density function at given x.
///  
/// # Examples
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate cf_dist_utils;
/// # fn main(){
/// let mu = 2.0;
/// let sigma = 5.0;
/// let num_u = 128;
/// let x = 0.0;
/// let x_min=-20.0;
/// let x_max=25.0;
/// let norm_cf_discrete=vec![Complex::new(1.0, 1.0), Complex::new(-1.0, 1.0)];
/// let cdf=cf_dist_utils::get_cdf_discrete_cf(
///     x, x_min, x_max, &norm_cf_discrete
/// );
/// # }
/// ```
pub fn get_cdf_discrete_cf(x: f64, x_min: f64, x_max: f64, discrete_cf: &[Complex<f64>]) -> f64 {
    fang_oost::get_expectation_single_element_real(x_min, x_max, x, discrete_cf, |u, x, u_index| {
        vk_cdf(u, x, x_min, u_index)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn var_works() {
        let mu = 2.0;
        let sigma = 5.0;
        let num_u = 128;
        let x_min = -20.0;
        let x_max = 25.0;
        let alpha = 0.05;
        let norm_cf = |u: &Complex<f64>| (u * mu + 0.5 * sigma * sigma * u * u).exp();
        let reference_var = 6.224268;
        let reference_es = 8.313564;
        let RiskMetric {
            expected_shortfall,
            value_at_risk,
        } = get_expected_shortfall_and_value_at_risk(
            alpha, num_u, x_min, x_max, 100, 0.0000001, &norm_cf,
        )
        .unwrap();
        assert_abs_diff_eq!(reference_var, value_at_risk, epsilon = 0.0001);
        assert_abs_diff_eq!(reference_es, expected_shortfall, epsilon = 0.001);
    }
    #[test]
    fn expectation_works() {
        let mu = 2.0;
        let sigma = 5.0;
        let num_u = 128;
        let x_min = -20.0;
        let x_max = 25.0;
        let norm_cf = |u: &Complex<f64>| (u * mu + 0.5 * sigma * sigma * u * u).exp();
        let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, norm_cf);
        let expected = get_expectation_discrete_cf(x_min, x_max, &discrete_cf);
        assert_abs_diff_eq!(expected, mu, epsilon = 0.0001);
    }
    #[test]
    fn error_works() {
        let mu = 2.0;
        let sigma = 5.0;
        let num_u = 128;
        let x_min = -20.0;
        let x_max = 25.0;
        let alpha = -0.5;
        let norm_cf = |u: &Complex<f64>| (u * mu + 0.5 * sigma * sigma * u * u).exp();
        let err = get_expected_shortfall_and_value_at_risk(
            alpha, num_u, x_min, x_max, 100, 0.0000001, &norm_cf,
        )
        .unwrap_err();
        assert_eq!(&err.to_string(), "Alpha needs to be within 0 and 1!");
    }
    #[test]
    fn variance_works() {
        let mu = 2.0;
        let sigma = 5.0;
        let num_u = 128;
        let x_min = -40.0;
        let x_max = 45.0;
        let norm_cf = |u: &Complex<f64>| (u * mu + 0.5 * sigma * sigma * u * u).exp();
        let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, norm_cf);
        let expected = get_variance_discrete_cf(x_min, x_max, &discrete_cf);
        assert_abs_diff_eq!(expected, sigma * sigma, epsilon = 0.0001);
    }
}
