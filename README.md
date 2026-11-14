# FOI Triage Desk (v8)

CDU IT Code Fair 2026, AI Challenge, Task 3: Information-access (FOI) request triage.

> How might we help an information officer find the records relevant to a request while guaranteeing a human reviews anything sensitive before it's released?

The desk finds the records responsive to an access request, flags passages that may fall under an exemption in the Information Act 2002 (NT), summarises the set, and hands every decision to the officer. **Jarvis**, a voice assistant built on Claude and ElevenLabs, helps the officer move through the work. It can navigate, search, explain and draft. It has no tool that can redact, release or sign off.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py            # builds data, evaluation and the interface (about 2 min)
python experiments/detector_cv.py   # optional: the cross-validation that chose the detector settings
python -m pytest -q tests    # 32 tests: the human gate, the workflow and the server
python server.py             # http://localhost:8000
```

Sign in with a demo account (password `demo`): **alex** (applicant), **jordan** (information officer) or **sam** (delegated decision-maker). Open two or three browser windows to work as different people on the same case at once.

## Roles and workflow

```
Applicant ──lodges──► AI triage ──► Information officer ──sends──► Decision-maker ──approves──► Applicant
 (alex)              retrieval +     reviews every flag            (sam)                       receives notice +
                     flagging        and every record   ◄─returns with a note─┘                redacted documents
                                     (jordan)
```

| Role | Can | Can't |
|---|---|---|
| Applicant | Lodge requests, follow a 5-step timeline, answer the officer's clarification questions, withdraw a request, read the decision notice and the released documents with redactions applied | See flags, AI output or internal notes; change anything |
| Information officer | Work the queue: scope, flag decisions, flag passages the AI missed, apply one decision to identical passages, record outcomes, flag settings; ask the applicant to clarify (pauses the 30-day clock); preview the release; lodge a request on someone's behalf; send for approval | Approve or release; change a case once it's sent |
| Delegated decision-maker | Review the officer's recommendation, every redaction and its reason; spot-check a sample of decisions; approve and release, or return with a note | Edit the review; approve a case they prepared |

The server enforces every rule (`foi_triage/workflow.py`), so going around the interface doesn't help. For example, an applicant calling the action API directly gets HTTP 403. The case store and its full audit trail live in `outputs/workflow.json`. The published page runs the same workflow inside one browser, where the three roles share that browser's data.

Jarvis works without keys, using built-in commands and the browser's own voice. For full AI and voice, copy `.env.example` to `.env` and add:

| Key | What it enables |
|---|---|
| `ANTHROPIC_API_KEY` | Jarvis understands free-form requests and uses the page's tools (Claude, tool use) |
| `ELEVENLABS_API_KEY` | Jarvis speaks with an ElevenLabs voice and transcribes speech with ElevenLabs Scribe |

Keys stay on the local server. The browser never sees them.

`outputs/officer_review.html` also opens directly in a browser. Jarvis then runs offline: Chrome and Edge can still take spoken commands.

## What's new in v7: answering the hard questions

| Weakness | What changed |
|---|---|
| Detector missed most new wording (26/73) | Person-name detection and word vectors (spaCy `en_core_web_md`), chosen by leave-phrasing-out cross-validation on **dev only**. Held-out: **60/73**, false alarms 22 → 20. On real email, names raise recall (5/22 → 9/22, core rules) with more false alarms; vectors stay off there because they flagged every dev email. Full table in `outputs/results.md`. |
| Names were only found by patterns | Named-entity recognition flags a person named in the body (not the record's sender or recipient, not greetings). Shown as "Names a person". |
| Officers might rubber-stamp | On sending, the server checks how the review was done: decisions under 4 s apart, or every flag getting the same decision, mark it as possibly rushed. The decision-maker must **spot-check a sample** before approving (every release despite a flag first, then high-harm grounds; 3, or 5 if rushed). Disagreeing blocks approval until the case is returned. |
| "Consult" was just a label | Consulting a third party is now a process: the party is pre-filled from the passage, a letter is drafted (download or copy), the reply is recorded (agrees / objects / no response) with a due date, and only then can the officer make the final decision. The record can't be confirmed, nor the case sent, while a consultation is open. The decision notice says how many third parties were consulted. |
| One annotator | `outputs/label_emails.html` is a blind labelling page for a teammate (keys Y / N / arrows, saves progress, downloads CSV). `python -m foi_triage.agreement second_labels.csv` reports Cohen's kappa, overall and per split, and lists every disagreement. |
| Bugs fixed | Overlapping redactions could leave text visible (a manual flag inside an AI flag): they now merge into one bar. Due dates were a day off in Australian time zones. |

## What's new in v6

- **Guided demo.** "Guided demo" on the sign-in page walks one request through all three people in 12 steps: Alex lodges it, Jordan reviews it, Sam approves it, Alex receives it. Each step switches account and explains what's happening. Built for the pitch.
- **Rule suggestions from officers.** Every passage an officer flags that the AI missed becomes a candidate rule on the Evaluation page, showing how many other records it would catch. `python -m foi_triage.feedback` prints the same list as Python lines for `sensitivity.RULES`. Nothing changes the detector until a person tests and adds it.
- **Downloads.** The applicant can save their decision and redacted documents as a printable HTML file, and the officer can save a draft preview; staff can save the full decision record as CSV and the rule suggestions as text. On the published page these go through the viewer's save prompt.
- **Fix:** Close buttons inside dialogs now work (a click handler was swallowing them).

## What's new in v5

- **Flag what the AI missed.** Select any text in a record, pick an exemption, and it becomes a flag that must be decided before the case can be sent. The server enforces this, and redactions carry through to the released documents.
- **Same passage, one decision.** When a flagged passage (a signature block, a phone number) appears in other records, the desk offers to apply the same decision to all of them.
- **Quick release reasons.** One-click reasons for releasing a flagged passage ("Applicant's own information", "Already public", and others), still editable.
- **Human–AI agreement.** The Evaluation view shows how often officers agreed with the AI's flags, overruled them, or added flags it missed, by category.

## What's in v3

1. **Real data** alongside the labelled synthetic corpus (see below).
2. **Jarvis voice agent:** Claude with 14 page tools, ElevenLabs text-to-speech and speech-to-text, push-to-talk, and an animated orb that follows the voice level.
3. **Neumorphic interface:** soft raised and pressed surfaces, light and dark themes, motion on view changes, record lists, flags, progress and the orb, with reduced-motion respected.
4. **Improvements:**
   - two collections
   - free-text "new request" search
   - a register of real NT requests
   - an evaluation view
   - keyboard shortcuts
   - a deadline ring
   - a server-side audit log with CSV export
   - session persistence
   - a decision-notice draft

## Real data

| Source | What it is | How it's used |
|---|---|---|
| **Enron emails**, [Enron-Spam](https://github.com/MWiechmann/enron_spam_data) (Metsis et al., 2006) | 16,545 real Enron employee emails released in the FERC investigation. We sample 1,500, undoing the dataset's lower-casing and tokenisation for readability. | Second collection: real, messy correspondence |
| **SARA** ([McKechnie, McDonald & Macdonald, SIGIR 2026](https://github.com/JackMcKechnie/SARA-A-Collection-of-Sensitivity-Aware-Relevance-Assessments)) | 50 real information needs about Enron, each with 3 crowdsourced queries. CC BY-NC 4.0. | Access requests on the Enron collection |
| **Right to Know** ([righttoknow.org.au](https://www.righttoknow.org.au/body/list/NT_state)) | 43 real public FOI requests to 8 NT bodies, with dates and outcomes | Request register, the parser test, and "Run" to send a real title into triage |
| **FOI disclosure logs** (from v2) | 11 real request texts | Parser test |
| **80 hand-labelled Enron emails** | Record-level labels: does the email hold material an FOI officer would need to review, and which kind | Real-text sensitivity check |

The labels on the 80 emails are from one annotator (Claude) and should be spot-checked by the team. SARA's own relevance judgements refer to the Berkeley-labelled Enron subset, whose text is hosted on Zenodo, which this build environment couldn't reach. Retrieval on Enron is therefore shown, not scored.

To regenerate the Enron sample, put `enron_spam_data.csv` (from the Enron-Spam repository) in `data/real/` and delete `data/real/enron_records.jsonl`. The prepared 1,500-email sample is already included.

## Results

Full tables are in `outputs/results.md`.

| | Result |
|---|---|
| NT retrieval: recall / precision / P@10 | 99% / 76% / 99% (v1: 78% / 22%) |
| NT request parsing: dates / agency / type | 100% / 100% / 100% |
| NT sensitive sentences, known wording | 184/184 |
| NT sensitive sentences, **new wording**, threshold 0.5 | **60/73** (v6: 26/73) |
| **Real emails, held-out 40**, threshold 0.5, core rules + names | 9/22 sensitive caught, 7/18 false alarms (v6: 5/22, 2/18) |
| **Real emails, held-out 40**, threshold 0.5, + correspondence pack | **14/22** caught, 11/18 false alarms (v6: 13/22, 8/18) |
| Real NT requests (Right to Know) | 12 of 43 long overdue, 5 of 43 fully or partly successful |

**What the numbers say.** On wording it was built from, the detector is near perfect. On new wording, and on real email from a different domain, it misses a lot. A small domain rule pack, written from 40 labelled emails and tested on 40 others, more than doubles recall at the cost of more false alarms. The design responds to that honestly:
- the system narrows and prioritises
- the officer reads and confirms every record
- `finalise()` refuses to run until they have

The real NT request outcomes show why faster triage matters.

## How Jarvis works

Jarvis is a different assistant for each person, with only the tools that person should have.

| Signed in as | Jarvis can | Jarvis can't |
|---|---|---|
| Applicant | Say where their requests are and what happens next, explain an exemption section in plain English, check the wording of a draft request, open the request form with suggested wording | See documents, flags or officer notes; lodge, withdraw or change anything |
| Information officer | Navigate, search, read and explain flags ("Why?" on each flag), say what's blocking sending, list consultations and open their letters, summarise, change the threshold, draft the decision notice | Redact, release, consult, confirm a record or send for approval |
| Decision-maker | List what's waiting, brief them on a case (releases despite a flag with reasons, consultations, review warnings, spot-check progress), open a case at its spot-checks | Approve, return, or spot-check; tell them whether to agree |

```
voice ──► browser speech recognition, or ElevenLabs Scribe via /api/stt
            │
            ▼
   Claude (tool use) ◄── role-specific system prompt and tool list
            │  calls only that role's page tools; anything else is refused in the page itself
            ▼
   reply ──► ElevenLabs TTS via /api/tts (or browser voice) ──► orb animates to the audio level
```

- **Hints.** When something needs attention, a one-line hint appears by the orb: a high-harm flag to decide, an overdue consultation, a case due within five days, a review marked as possibly rushed, or an applicant's unanswered question. Clicking it asks Jarvis about it; × dismisses it.
- **Control.** A stop button cancels a reply in progress; the bin icon clears the conversation. Signing in as someone else starts a fresh conversation.
- **Three runtimes, one interface:** the local server (Claude API and ElevenLabs, keys server-side), the published page (its built-in Claude access with the same tools; spoken replies in the browser voice), and offline (pattern-matched commands for each role).
- **Safety by construction.** No role has a tool that decides anything. Everything Jarvis does for the officer is logged as "Jarvis (for officer)".
- **Voice.** The default is the stock ElevenLabs voice "George" (`JBFqnCBsd6RMkjVDRZzb`). Don't clone a real person's voice without consent.

## Keyboard

| Key | Action |
|---|---|
| `J` | Open or close Jarvis |
| `M` | Talk to Jarvis |
| `R` / `E` / `C` | Redact / release (asks for a reason) / consult the current flag |
| `N` | Next record with open flags |
| `↑` / `↓` | Previous / next record |
| `Esc` | Close panels |

## Pipeline

1. **Parse:** dates, agency, document type and topic from the request text.
2. **Retrieve:** BM25 keyword matching with synonyms, fused with LSA semantic similarity. Dates and document type are hard filters; agency only adjusts the score.
3. **Flag:** per sentence, rules plus a classifier, giving a confidence and reasons. Health, cultural and law enforcement flags are always reviewed from 0.25. Optional domain rule packs.
4. **Summarise:** extractive, citing records; flagged sentences are withheld until decided.
5. **Human gate:** `ReviewSession` blocks `finalise()` until every flag has a decision and every record has an outcome.

Exemption categories map to s45, s46, s49(d), s52, s56 (privacy, health, cultural) and s57. s52, s56 and s57 also need the s50 public interest test, which only the officer applies. The response deadline is 30 days (s19).

## Files

```
run_all.py                  end-to-end build → outputs/
experiments/detector_cv.py  dev-only cross-validation that chose the detector settings
server.py                   local server: UI, sign-in, cases API, /api/claude, /api/tts, /api/stt
foi_triage/
  corpus.py                 synthetic NT corpus + answer key
  real_data.py              Enron, SARA and Right to Know loaders
  exemptions.py             categories mapped to the Act
  retrieval.py              request parsing, hybrid retrieval
  sensitivity.py            rules + classifier + domain packs, routing
  summarise.py              extractive summary
  review.py                 ReviewSession, ReleaseBlocked, audit
  pipeline.py               packets per collection
  feedback.py               officer-added flags → candidate rules
  agreement.py              second annotator: labelling page + Cohen's kappa
  workflow.py               roles, case statuses, gates, release package, decision notice
  evaluate.py               retrieval, sensitivity, threshold sweep, real-email check
  ui.py                     embeds data into the interface
ui/template.html            interface + Jarvis
data/                       synthetic corpus; data/real/ real datasets and labels
tests/                      gate, workflow and server tests (Claude and ElevenLabs mocked)
outputs/                    generated results, packets and interface
```

## Limits

- The real-email labels have one annotator until a teammate completes `label_emails.html`; quote kappa alongside the email results.
- spaCy's name recognition sometimes tags companies as people ("Dixon-Dean asked that its rates…"). The officer sees the reason and decides.
- Cultural-information rules need review with Land Council or Aboriginal Interpreter Service advisers.
- Rule suggestions are literal and can be broad (check the "would also match" count). Feeding officer flags back as classifier training labels is the next step.
- Thresholds and the miss-versus-review cost should be set with information officers.
