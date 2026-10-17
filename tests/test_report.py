from loanshield.report import fusion_name, numbering, parse, pm, render, segments, signed

TEMPLATE = """<!-- comment with <<not.a.key>> -->
# Results

See @tab:x and @fig:y. Kappa <<data.kappa>>, PR <<main.rf>> on *organic* apps.

::: table {#tab:x}
caption: A table.
| Model | PR-AUC |
| RF | <<main.rf>> |
| --- | --- |
| LR | <<main.lr>> |
:::

::: figure {#fig:y}
caption: A figure.
image: does_not_exist.png 0.5
:::

[[AUTHOR: write this.]]
"""


def test_formatting_helpers():
    assert pm(0.6123, 0.18) == "0.61 ± 0.18"
    assert signed(-0.0421) == "−0.042" and signed(0.01, 2) == "+0.01"
    assert fusion_name("fusion(rf+tfidf_svm)") == "Random forest + TF-IDF + SVM"


def test_parse_blocks_and_numbering():
    blocks = parse(TEMPLATE)
    assert [b.kind for b in blocks] == ["heading1", "para", "table", "figure", "author"]
    table = blocks[2]
    assert table.caption == "A table." and len(table.rows) == 4  # header, RF, rule, LR
    assert numbering(blocks) == {"tab:x": "Table 1", "fig:y": "Fig. 1"}


def test_segments_fill_refs_and_mark_missing():
    missing = []
    segs = segments("See @tab:x, PR <<main.rf>>, kappa <<data.kappa>> on *organic*.",
                    {"main.rf": "0.61 ± 0.18"}, {"tab:x": "Table 1"}, missing)
    assert ("normal", "Table 1") in segs and ("normal", "0.61 ± 0.18") in segs
    assert ("missing", "[data.kappa]") in segs and ("italic", "organic") in segs
    assert missing == ["data.kappa"]


def test_render_produces_docx_text():
    doc, missing = render(TEMPLATE, {"main.rf": "0.61 ± 0.18", "main.lr": "0.50 ± 0.10"})
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "See Table 1 and Fig. 1." in text and "[data.kappa]" in text
    assert "[AUTHOR: write this.]" in text and "not.a.key" not in text
    cells = [c.text for c in doc.tables[0]._cells]
    assert cells == ["Model", "PR-AUC", "RF", "0.61 ± 0.18", "LR", "0.50 ± 0.10"]
    assert "data.kappa" in missing and "figure:does_not_exist.png" in missing


def test_bold_and_italic_runs():
    segs = segments("**Head.** body *it* end", {}, {}, [])
    assert ("bold", "Head.") in segs and ("italic", "it") in segs and not any("*" in t for _, t in segs)
