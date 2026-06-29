use super::*;

#[test]
fn new_sets_total() {
    let usage = Usage::new(10, 5);
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

#[test]
fn add_accumulates_all_fields() {
    let a = Usage {
        input_tokens: 1,
        output_tokens: 2,
        total_tokens: 3,
        cache_read_tokens: 4,
        cache_creation_tokens: 5,
        reasoning_tokens: 6,
    };
    let b = a;
    let sum = a + b;
    assert_eq!(sum.input_tokens, 2);
    assert_eq!(sum.cache_read_tokens, 8);
    assert_eq!(sum.reasoning_tokens, 12);
}

#[test]
fn add_assign_works() {
    let mut total = Usage::default();
    total += Usage::new(3, 4);
    total += Usage::new(1, 1);
    assert_eq!(total.total_tokens, 9);
}

#[test]
fn effective_total_falls_back() {
    let usage = Usage {
        input_tokens: 4,
        output_tokens: 6,
        total_tokens: 0,
        ..Usage::default()
    };
    assert_eq!(usage.effective_total(), 10);
}

#[test]
fn usage_totals_count_calls() {
    let mut totals = UsageTotals::new();
    totals.record(Usage::new(10, 10));
    totals += Usage::new(5, 5);
    assert_eq!(totals.calls, 2);
    assert_eq!(totals.usage.total_tokens, 30);
}
