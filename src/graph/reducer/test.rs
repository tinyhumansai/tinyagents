//! Unit tests for the built-in reducers (overwrite, append, set-union, min,
//! max) and the closure-backed channel/state reducers.

use super::*;

#[test]
fn overwrite_keeps_update() {
    let r = OverwriteReducer;
    assert_eq!(Reducer::reduce(&r, 1, 2).unwrap(), 2);
}

#[test]
fn append_concatenates() {
    let r = AppendReducer;
    assert_eq!(r.reduce(vec![1, 2], vec![3, 4]).unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn set_union_dedups() {
    let r = SetUnionReducer;
    assert_eq!(r.reduce(vec![1, 2], vec![2, 3]).unwrap(), vec![1, 2, 3]);
}

#[test]
fn min_and_max() {
    assert_eq!(Reducer::reduce(&MinReducer, 5, 3).unwrap(), 3);
    assert_eq!(Reducer::reduce(&MinReducer, 2, 9).unwrap(), 2);
    assert_eq!(Reducer::reduce(&MaxReducer, 5, 3).unwrap(), 5);
    assert_eq!(Reducer::reduce(&MaxReducer, 2, 9).unwrap(), 9);
}

#[test]
fn closure_reducer_runs() {
    let r = ClosureReducer::new(|a: i32, b: i32| Ok(a + b));
    assert_eq!(r.reduce(2, 3).unwrap(), 5);
}

#[test]
fn closure_state_reducer_applies_partial() {
    #[derive(Clone, PartialEq, Debug)]
    struct S {
        total: i32,
    }
    let r = ClosureStateReducer::new(|mut s: S, u: i32| {
        s.total += u;
        Ok(s)
    });
    let s = r.apply(S { total: 1 }, 4).unwrap();
    assert_eq!(s.total, 5);
}

#[test]
fn overwrite_state_reducer() {
    let r = OverwriteStateReducer;
    assert_eq!(StateReducer::apply(&r, 1, 2).unwrap(), 2);
}
