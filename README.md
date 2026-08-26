# cTMT Digital Biomarker — Streamlit deployment

연구용 cTMT 마우스 궤적 디지털 바이오마커 프로토타입입니다.

## Pipeline

cTMT → Mouse X/Y/Timestamp → 103 Digital Features → SelectKBest(20) → StandardScaler → SVC → Research behavioral-pattern score

## Research basis

- Public study dataset: 74 participants (Control 33 / MCI 41)
- Digital input features: 103
- Nested-CV reproduction ROC-AUC: 0.6704
- Raw → 103-feature reproduction: 74/74 participants matched the published processed dataset to numerical precision

## Local run

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Upload the **contents of this folder** to the root of a GitHub repository.
2. Open Streamlit Community Cloud and create an app from that repository.
3. Set Main file path to `app.py`.
4. In Advanced settings, select **Python 3.13** to match the model-development environment.
5. No secrets or `packages.txt` are required.
6. Deploy and complete one full research-mode cTMT session as an end-to-end check.

## Important interpretation note

This is a research/education prototype, not a medical device. The displayed `MCI 관련 행동패턴 점수` is the model's class output and must not be interpreted as an individual's true probability of MCI or as a clinical diagnosis.
