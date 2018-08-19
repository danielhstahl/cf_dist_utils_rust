//! Contains functions for computing the partial expectation, quantile, and cumulative density 
//! function given a characteristic function.  
extern crate fang_oost;
extern crate rootfind;
extern crate num_complex;
extern crate rayon;
#[macro_use]
#[cfg(test)]
extern crate approx;

use num_complex::Complex;
use rootfind::solver::bisection;
use rootfind::bracket::{Bounds};
use rootfind::wrap::RealFn;
use rayon::prelude::*;
/**
    Function to compute the CDF of a distribution; see 
    http://danielhstahl.com/static/media/CreditRiskExtensions.c31991d2.pdf
    
*/
fn vk_cdf(
    u:f64,
    x:f64,
    a:f64,
    k:usize
)->f64{
    if k==0 {x-a} else {((x-a)*u).sin()/u}
}

fn diff_pow(x:f64, a:f64)->f64{
    0.5*(x.powi(2)-a.powi(2))
}

/**
    Function to compute the partial expectation of a distribution; see
    http://danielhstahl.com/static/media/CreditRiskExtensions.c31991d2.pdf.
*/
fn vk_pe(
    u:f64,
    x:f64,
    a:f64,
    k:usize
)->f64 {
    let arg=(x-a)*u;
    let u_den=1.0/u;
    if k==0 {
        diff_pow(x, a)
    } 
    else {
        x*arg.sin()*u_den+u_den.powi(2)*(arg.cos()-1.0)
    }
}

fn compute_value_at_risk(
    alpha:f64,
    x_min:f64,
    x_max:f64,
    discrete_cf:&[Complex<f64>]
)->f64

{
    let bounds=Bounds::new(x_min, x_max);
    let vf=|u, x, u_index|{
        vk_cdf(u, x, x_min, u_index)
    };
    let in_f=|x:f64|{
        fang_oost::get_expectation_single_element_real(
            x_min, x_max, x, 
            &discrete_cf, vf
        )-alpha
    };
    let f=RealFn::new(&in_f);
    -bisection(&f, &bounds, 100).unwrap()
}
fn compute_expected_shortfall(
    alpha:f64,
    x_min:f64,
    x_max:f64,
    value_at_risk:f64,
    discrete_cf:&[Complex<f64>]
)->f64{
    -fang_oost::get_expectation_single_element_real(
        x_min, x_max, -value_at_risk, 
        discrete_cf, 
        |u, x, u_index|{
            vk_pe(u, x, x_min, u_index)
        }
    )/alpha
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
/// let alpha=0.05;
/// let norm_cf=|u:&Complex<f64>| (u*mu+0.5*sigma*sigma*u*u).exp();
/// let reference_var=6.224268;
/// let reference_es=8.313564;
/// let (estimated_es, estimated_var)=cf_dist_utils::get_expected_shortfall_and_value_at_risk(
///     alpha, num_u, x_min, x_max, norm_cf
/// );
/// assert_abs_diff_eq!(reference_var, estimated_var, epsilon=0.0001);
/// assert_abs_diff_eq!(reference_es, estimated_es, epsilon=0.001);
/// # }
/// ```
pub fn get_expected_shortfall_and_value_at_risk<T>(
    alpha:f64,
    num_u:usize,
    x_min:f64,
    x_max:f64,
    fn_inv:T
)->(f64, f64)
where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discrete_cf=fang_oost::get_discrete_cf(num_u, x_min, x_max, fn_inv);
    let value_at_risk=compute_value_at_risk(
        alpha, x_min, 
        x_max, &discrete_cf
    );
    let expected_shortfall=compute_expected_shortfall( 
        alpha, x_min, 
        x_max, 
        value_at_risk, 
        &discrete_cf
    );
    (expected_shortfall, value_at_risk)
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
/// let reference_cdf=0.7257469;
/// let cdf=cf_dist_utils::get_cdf(
///     num_x, num_u, x_min, x_max, &norm_cf
/// );
/// # }
/// ```
pub fn get_cdf<T>(
    num_x:usize,
    num_u:usize,
    x_min:f64,
    x_max:f64,
    cf:T
)->Vec<f64> 
where T:Fn(&Complex<f64>)->Complex<f64>
+std::marker::Sync+std::marker::Send
{
    fang_oost::get_expectation_x_real(
        num_x, num_u, 
        x_min, x_max, cf, 
        |u, x, u_index|{
            vk_cdf(u, x, x_min, u_index)
        }
    ).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn var_works() {
        let mu=2.0;
        let sigma=5.0;
        let num_u=128;
        let x_min=-20.0;
        let x_max=25.0;
        let alpha=0.05;
        let norm_cf=|u:&Complex<f64>| (u*mu+0.5*sigma*sigma*u*u).exp();
        let reference_var=6.224268;
        let reference_es=8.313564;
        let (estimated_es, estimated_var)=get_expected_shortfall_and_value_at_risk(
            alpha, num_u, x_min, x_max, &norm_cf
        );
        assert_abs_diff_eq!(reference_var, estimated_var, epsilon=0.0001);
        assert_abs_diff_eq!(reference_es, estimated_es, epsilon=0.001);
    }
}
