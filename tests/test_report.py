from loanshield.report import fusion_name, pm, render, signed, tex


def test_tex_escapes_specials():
    assert tex("A&B 50% $x_1 #2 {y}") == r"A\&B 50\% \$x\_1 \#2 \{y\}"


def test_render_fills_and_marks_missing():
    out, missing = render("PR <<main.organic.rf.pr_auc>>, kappa <<data.kappa>>.",
                          {"main.organic.rf.pr_auc": pm(0.6123, 0.18)})
    assert out == r"PR $0.61 \pm 0.18$, kappa \MISSING{data.kappa}."
    assert missing == ["data.kappa"]


def test_formatting_helpers():
    assert signed(-0.0421) == "$-0.042$" and signed(0.01, 2) == "$+0.01$"
    assert fusion_name("fusion(rf+tfidf_svm)") == "Random forest + TF-IDF + SVM"
