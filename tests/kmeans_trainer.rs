extern crate opencv;
use opencv::core;
use opencv::core::{Mat, Size, TermCriteria};
use opencv::features2d;

use opencv::features2d::{BOWTrainer, BOWKMeansTrainer};

fn create_test_data(n: i32) -> Result<Mat, String> {
    let mut mat = Mat::new_rows_cols(n, 2, core::CV_32FC1)?;
    let size = mat.size()?;
    assert_eq!(core::Size { width: 2, height: n }, size);
    assert_eq!(core::CV_32F, mat.depth()?);
    assert_eq!(1, mat.channels()?);

    let minus_one_scalar = core::Scalar{ data: [-1f64, 0., 0., 0.] };
    let one_scalar = core::Scalar{ data: [1f64, 0., 0., 0.] };
    let minus_one = Mat::new_rows_cols_with_default(1, 1, core::CV_32FC1, minus_one_scalar)?;
    let one = Mat::new_rows_cols_with_default(1, 1, core::CV_32FC1, one_scalar)?;

    let no_mask = Mat::new()?;
    for i in 0..n {
        let row = mat.row(n)?; // panics here...
        let v = if i % 2 == 1 { &one } else { &minus_one };
        let mut cell = row.col(0)?;
        cell.set_to(v, &one)?;
        let v = if (i>>1) % 2 == 1 { &one } else { &minus_one };
        let mut cell = row.col(1)?;
        cell.set_to(v, &one)?;
    }

    Ok(mat)
}

#[test]
fn kmeans_trainer_cluster() {
    const N: i32 = 100;
    const CLUSTER_COUNT: i32 = 4;

    let mat = create_test_data(N).unwrap();

    let criteria = TermCriteria::new(
        core::TermCriteria_MAX_ITER | core::TermCriteria_EPS, 100, 1e-3).unwrap();
    let mut trainer = BOWKMeansTrainer::new_with_criteria(
        CLUSTER_COUNT, // clusterCount: i32,
        &criteria, // termcrit: &TermCriteria,
        5, // attempts: i32,
        core::KMEANS_PP_CENTERS  // flags: i32)
    ).unwrap();

    // TODO

    trainer.add(&mat).unwrap();
    
    let outcome = trainer.cluster().unwrap();
    assert_eq!(core::CV_32F, outcome.depth().unwrap());
    let outcome_size = outcome.size().unwrap();
    assert_eq!(outcome_size, core::Size{ width: 2, height: CLUSTER_COUNT});

    let descriptors = trainer.get_descriptors().unwrap();
    assert_eq!(descriptors.len() as usize, CLUSTER_COUNT as usize);
}

#[test]
fn kmeans_trainer_cluster_with() {
    const N: i32 = 100;
    const CLUSTER_COUNT: i32 = 4;

    let mat = create_test_data(N).unwrap();

    let criteria = TermCriteria::new(
        core::TermCriteria_MAX_ITER | core::TermCriteria_EPS, 100, 1e-3).unwrap();
    let trainer = BOWKMeansTrainer::new_with_criteria(
        CLUSTER_COUNT, // clusterCount: i32,
        &criteria, // termcrit: &TermCriteria,
        5, // attempts: i32,
        core::KMEANS_PP_CENTERS  // flags: i32)
    ).unwrap();

    // TODO
    let outcome = trainer.cluster_with(&mat).unwrap();
    assert_eq!(core::CV_32F, outcome.depth().unwrap());
    let outcome_size = outcome.size().unwrap();
    assert_eq!(outcome_size, core::Size{ width: 2, height: CLUSTER_COUNT});
}
