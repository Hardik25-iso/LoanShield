from loanshield.label.match import distinctive, name_similarity, normalise, offers_credit


def test_normalise_strips_taglines_and_suffixes():
    assert normalise("Prefr: Online Loan App India", legal=False) == "prefr"
    assert normalise("Branch- Personal Cash Loan App", legal=False) == "branch personal cash"
    assert normalise("DreamPlug Technologies Pvt Ltd") == "dreamplug"
    assert normalise("App Store: Axis Mobile: Pay, Invest & UPI", legal=False) != ""


def test_generic_tokens_do_not_create_false_matches():
    # token_set_ratio alone would score 100 here (subset of tokens)
    assert name_similarity(normalise("Cash"), normalise("Fast Cash")) < 80
    assert name_similarity(normalise("Kreditbee", legal=False),
                           normalise("KreditBee: Personal Loan App", legal=False)) == 100
    # token subsets must not count as matches (seen in the dry run)
    assert name_similarity(normalise("Gaadi Bazaar", legal=False),
                           normalise("My Money Bazaar", legal=False)) < 80
    assert name_similarity(normalise("Union Bank of India"), normalise("City Union Bank Limited")) < 92


def test_distinctive():
    assert distinctive("fast cash loan") == ""
    assert distinctive("moneyview personal loan") == "moneyview"


def test_offers_credit():
    desc = "Get a personal loan up to 5 lakh. Interest rate 18% APR, tenure 3-24 months. Disbursal in minutes."
    assert offers_credit("QuickRupee", "Instant loan", desc)[0] == "yes"
    assert offers_credit("Loan EMI Calculator", "", "interest rate and tenure")[0] == "no"
    assert offers_credit("Daily Horoscope", "", "Your stars today")[0] == "no"
    assert offers_credit("Payments", "", "Pay bills. Personal loan coming soon")[0] == "unsure"
    assert offers_credit("X", "", "Celebrate April with us")[0] == "no"  # 'apr' must not match 'april'
