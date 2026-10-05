"""Generates the demo documents in samples/.

The content is invented. It is shaped like a clinical study report synopsis so
that the assistant can be tried on realistic questions (endpoints, populations,
adverse events) without using any real trial or patient data.

    python scripts/make_sample_docs.py
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent.parent / "samples"
BANNER = "SYNTHETIC DOCUMENT - written for software testing. Not real clinical data."

PAGES: list[tuple[str, list[str]]] = [
    (
        "Clinical Study Report Synopsis: Study VLT-201 (ASCENT-HTN)",
        [
            "Sponsor: Northlake Biopharma. Investigational product: veltaprin (VLT-118), "
            "film-coated tablets for oral use, taken once daily in the morning.",
            "Veltaprin is a selective aldosterone synthase inhibitor. It lowers aldosterone "
            "production by inhibiting the CYP11B2 enzyme, with low affinity for CYP11B1, the "
            "enzyme responsible for cortisol synthesis.",
            "Indication studied: uncontrolled hypertension in adults who are already receiving "
            "three or more antihypertensive medicines.",
            "Study title: A Phase 2b, randomized, double-blind, placebo-controlled, "
            "dose-ranging, parallel-group study of the efficacy and safety of veltaprin in "
            "adults with uncontrolled hypertension.",
            "The study was conducted at 42 sites in 6 countries (Germany, Poland, Spain, "
            "Switzerland, the United Kingdom and the United States). The first participant "
            "was enrolled on 14 March 2022 and the last participant completed the last visit "
            "on 29 September 2023.",
        ],
    ),
    (
        "1. Objectives and study design",
        [
            "Primary objective: to evaluate the effect of veltaprin compared with placebo on "
            "the change from baseline in mean seated systolic blood pressure (SBP) at week 12. "
            "This change was the primary endpoint of the study.",
            "Secondary endpoints: change from baseline in mean seated diastolic blood pressure "
            "(DBP) at week 12; proportion of participants with seated SBP below 130 mmHg at "
            "week 12; change in 24-hour ambulatory SBP in a substudy; safety and tolerability.",
            "Design: after a 2-week single-blind placebo run-in period, eligible participants "
            "were randomized in a 1:1:1:1 ratio to placebo or veltaprin 2.5 mg, 5 mg or 10 mg "
            "once daily for 12 weeks, followed by a 4-week safety follow-up without study "
            "treatment. Randomization was stratified by baseline eGFR (below 60 versus 60 or "
            "above mL/min/1.73 m2) and by type 2 diabetes status.",
            "Key inclusion criteria: age 18 to 80 years; mean seated SBP of at least 140 mmHg "
            "and below 180 mmHg while on stable doses of three or more antihypertensive "
            "medicines, one of them a diuretic, for at least 8 weeks.",
            "Key exclusion criteria: serum potassium above 5.0 mmol/L at screening; eGFR below "
            "45 mL/min/1.73 m2; myocardial infarction or stroke within the previous 6 months; "
            "New York Heart Association class III or IV heart failure; pregnancy or "
            "breastfeeding.",
        ],
    ),
    (
        "2. Study population",
        [
            "A total of 612 people were screened and 412 participants were randomized: 103 to "
            "placebo, 102 to veltaprin 2.5 mg, 104 to veltaprin 5 mg and 103 to veltaprin "
            "10 mg. All randomized participants received at least one dose of study treatment.",
            "Overall, 381 participants (92.5%) completed the 12-week treatment period. "
            "Thirty-one participants discontinued early: 12 because of adverse events, 9 "
            "withdrew consent, 6 were lost to follow-up and 4 discontinued for other reasons.",
            "Baseline characteristics were balanced across groups. The mean age was 61.4 "
            "years and 46% of participants were female. Mean seated blood pressure at "
            "baseline was 152.8/88.1 mmHg. Type 2 diabetes was present in 38% of participants "
            "and the mean eGFR was 71 mL/min/1.73 m2. Participants were taking a mean of 3.4 "
            "antihypertensive medicines at baseline.",
        ],
    ),
    (
        "3. Efficacy results",
        [
            "Primary endpoint. The least-squares mean change from baseline in seated SBP at "
            "week 12 was -4.1 mmHg with placebo, -9.8 mmHg with veltaprin 2.5 mg, -14.6 mmHg "
            "with 5 mg and -17.9 mmHg with 10 mg.",
            "The placebo-adjusted differences were -5.7 mmHg for 2.5 mg (95% CI -9.2 to -2.2; "
            "p=0.002), -10.5 mmHg for 5 mg (95% CI -14.0 to -7.0; p<0.001) and -13.8 mmHg for "
            "10 mg (95% CI -17.3 to -10.3; p<0.001). The analysis used a mixed model for "
            "repeated measures. The primary endpoint was met for all three doses.",
            "Secondary endpoints. The mean change in seated DBP at week 12 was -1.9 mmHg with "
            "placebo and -4.2, -6.8 and -8.3 mmHg with veltaprin 2.5 mg, 5 mg and 10 mg.",
            "The proportion of participants with seated SBP below 130 mmHg at week 12 was "
            "14.6% with placebo, 27.5% with 2.5 mg, 41.3% with 5 mg and 49.5% with 10 mg.",
            "In the ambulatory blood pressure substudy (168 participants), the "
            "placebo-adjusted change in 24-hour SBP with veltaprin 10 mg was -11.2 mmHg. "
            "Results were consistent across subgroups defined by baseline eGFR and diabetes "
            "status.",
        ],
    ),
    (
        "4. Safety results",
        [
            "Treatment-emergent adverse events were reported in 38.8% of participants on "
            "placebo and in 41.2%, 44.2% and 49.5% of participants on veltaprin 2.5 mg, 5 mg "
            "and 10 mg.",
            "Hyperkalemia, defined as serum potassium above 5.5 mmol/L, was the adverse event "
            "of special interest. It occurred in 1 participant (1.0%) on placebo, 3 (2.9%) on "
            "2.5 mg, 6 (5.8%) on 5 mg and 10 (9.7%) on 10 mg. Five participants discontinued "
            "treatment because of hyperkalemia: 1 in the 5 mg group and 4 in the 10 mg group.",
            "Dizziness was reported in 2.9% of participants on placebo and in 3.9%, 5.8% and "
            "6.8% on veltaprin 2.5 mg, 5 mg and 10 mg. Headache and fatigue occurred at "
            "similar rates in all groups.",
            "Serious adverse events were reported in 9 participants (2.2%) overall. One "
            "serious event, hyperkalemia requiring hospitalization in the 10 mg group, was "
            "considered related to study treatment. No deaths occurred during the study.",
            "Morning serum cortisol did not change in a clinically meaningful way and ACTH "
            "stimulation tests remained normal in all dose groups. The mean eGFR decreased by "
            "3.2 mL/min/1.73 m2 at week 12 in the 10 mg group and returned to baseline during "
            "follow-up.",
        ],
    ),
    (
        "5. Conclusions",
        [
            "Veltaprin given once daily for 12 weeks produced dose-dependent, clinically "
            "relevant reductions in systolic blood pressure in adults with uncontrolled "
            "hypertension on three or more medicines.",
            "The 5 mg once-daily dose was selected for Phase 3. It delivered most of the "
            "blood pressure reduction seen with 10 mg while the rate of hyperkalemia was "
            "lower (5.8% versus 9.7%).",
            "The Phase 3 program will include serum potassium monitoring at weeks 1, 4 and 12.",
            "Limitations: treatment lasted 12 weeks, so durability of the effect is not "
            "known; 11% of participants were Black, which limits conclusions for this group; "
            "the study was not designed to assess cardiovascular outcomes.",
        ],
    ),
]

SITE_MEMO = """\
SYNTHETIC DOCUMENT - written for software testing. Not real clinical data.

Study VLT-201 (ASCENT-HTN) - Memo to investigational sites
Subject: Protocol amendment 2, potassium monitoring
Date: 6 June 2022

Background
The independent data monitoring committee reviewed unblinded safety data after the first 120 participants had completed 4 weeks of treatment. The committee recommended that the study continue and asked for closer monitoring of serum potassium.

What changes
An additional serum potassium measurement is required at week 1 (day 7, with a window of plus or minus 2 days). Until now the first post-baseline measurement was at week 2.

Study treatment must be interrupted if serum potassium is 5.6 mmol/L or higher. The measurement must be repeated within 72 hours. Treatment may be restarted only if the repeat value is below 5.0 mmol/L.

Study treatment must be stopped permanently if serum potassium is 6.0 mmol/L or higher at any time, or if treatment has been interrupted twice for hyperkalemia.

Participants should be advised not to start potassium supplements or potassium-containing salt substitutes during the study.

What does not change
Dosing, visit schedule after week 2 and all efficacy assessments remain as described in protocol version 2.0.

Questions about this memo go to the study medical monitor through the usual site contact.
"""


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, BANNER, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"VLT-201 CSR synopsis - page {self.page_no()}", align="C")


def build_pdf(path: Path) -> None:
    pdf = Report(format="A4")
    pdf.set_margins(22, 18, 22)
    pdf.set_auto_page_break(auto=True, margin=20)
    for title, paragraphs in PAGES:
        pdf.add_page()
        pdf.set_text_color(20)
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 11)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 6, paragraph, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
    pdf.output(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_pdf(OUT / "vlt201_csr_synopsis.pdf")
    (OUT / "vlt201_site_memo_potassium.txt").write_text(SITE_MEMO, encoding="utf-8")
    print(f"wrote {len(PAGES)}-page PDF and site memo to {OUT}")
