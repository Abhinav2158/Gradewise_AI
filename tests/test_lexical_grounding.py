import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.text_align import compute_lexical_grounding

def test_full_keyword_coverage():
    span = "The reaction produces carbon dioxide gas which escapes into the air."
    condition = "Explains that carbon dioxide gas escapes"
    score = compute_lexical_grounding(span, condition)
    # Tokens in condition: {'explains', 'carbon', 'dioxide', 'gas', 'escapes'}
    # Tokens in span: {'reaction', 'produces', 'carbon', 'dioxide', 'gas', 'escapes', 'air'}
    # Matched: {'carbon', 'dioxide', 'gas', 'escapes'} -> 4/5 = 0.80
    assert score >= 0.75

def test_zero_keyword_coverage():
    span = "The temperature increased rapidly during the test."
    condition = "States that mass is conserved in a sealed flask"
    score = compute_lexical_grounding(span, condition)
    assert score == 0.0

def test_target_keywords_list():
    span = "The student computed mean 42 and standard deviation 8."
    keywords = ["mean", "42", "deviation", "8"]
    score = compute_lexical_grounding(span, "", target_keywords=keywords)
    assert score == 1.0

def test_empty_input_handling():
    assert compute_lexical_grounding("", "some condition") == 0.0
    assert compute_lexical_grounding("some span", "") == 0.0

if __name__ == "__main__":
    test_full_keyword_coverage()
    test_zero_keyword_coverage()
    test_target_keywords_list()
    test_empty_input_handling()
    print("All lexical grounding tests passed successfully!")
