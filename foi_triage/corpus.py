"""Synthetic NT Government record corpus with an answer key.

What changed from v1
--------------------
* Real English sentences per topic instead of lorem ipsum, so retrieval has
  to work on vocabulary, not on the topic label repeated in the subject.
* Eight exemption categories (adds health, Aboriginal cultural information
  and legal privilege, which matter most in the NT).
* Each category has several phrasings. Some phrasings are HELD OUT: they are
  only used in "test" records. The rules and the classifier are built from
  the "dev" phrasings only, so test scores measure whether detection works
  on wording it has not seen. v1 scored 100% because rules and data came
  from the same templates.
* Decoy sentences that look sensitive but are not (a published contract
  value, the Minister's media release, a kitchen cabinet). These measure
  over-flagging honestly.
* Sentence-level span labels (character offsets) instead of one flag per
  record, so we can count exactly which items were caught or missed.
* Responsiveness labels are deterministic (topic + date range + agency or cc
  + document type), not random.

Everything is fictional and generated locally with a fixed seed.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

SEED = 2026
fake = Faker("en_AU")

AGENCIES = {
    "Health": "NT Department of Health",
    "Housing": "NT Department of Housing, Local Government and Community Development",
    "Police": "NT Police, Fire and Emergency Services",
    "Infrastructure": "NT Department of Infrastructure, Planning and Logistics",
    "Treasury": "NT Treasury",
}

NT_PLACES = ["Darwin", "Palmerston", "Katherine", "Alice Springs", "Tennant Creek",
             "Nhulunbuy", "Casuarina", "Humpty Doo", "Jabiru"]
COMMUNITIES = ["Maningrida", "Wadeye", "Ngukurr", "Gunbalanya", "Yuendumu",
               "Borroloola", "Galiwinku", "Lajamanu", "Ramingining"]
FACILITIES = ["Katherine depot", "Casuarina office", "Alice Springs store",
              "Palmerston works yard", "Tennant Creek clinic"]
DOC_TYPES = ["Email", "Briefing note", "Memo", "Meeting minutes", "Report"]

# ---------------------------------------------------------------------------
# Topics: owning agency, request phrasings, and ordinary content sentences
# ---------------------------------------------------------------------------
TOPICS = {
    "remote housing maintenance backlog": {
        "agency": "Housing",
        "phrasings": ["remote housing maintenance backlog", "repair delays for houses in remote communities",
                      "the backlog of repairs to remote community housing"],
        "sentences": [
            "There are {n3} open repair jobs across remote communities, with {n2} marked urgent.",
            "Plumbing and electrical faults make up most of the outstanding maintenance work in {community}.",
            "Average time to complete an urgent repair in {community} is {n2} days against a two-day target.",
            "Travel to {community} accounts for a large share of the cost of each maintenance visit.",
            "The regional trades hub trial aims to clear the repair backlog by the end of the dry season.",
            "Contractors reported that blocked sewers and broken hot water systems are the most common faults.",
            "Housing officers will inspect overdue repair jobs in {community} next month.",
        ],
    },
    "alcohol management plan review": {
        "agency": "Health",
        "phrasings": ["the alcohol management plan review", "reviews of community alcohol management plans",
                      "alcohol management plans and their review"],
        "sentences": [
            "The {community} alcohol management plan is due for its three-year review.",
            "Community consultations on the alcohol plan were held over two days in {community}.",
            "Takeaway alcohol sales data was compared before and after the plan took effect.",
            "The review panel will report on whether the alcohol plan reduced harm in {community}.",
            "Stakeholders asked for the alcohol management plan to include more support services.",
            "Draft terms of reference for the alcohol plan review were circulated to the reference group.",
        ],
    },
    "liquor licensing compliance": {
        "agency": "Police",
        "phrasings": ["liquor licensing compliance", "compliance checks on liquor licences",
                      "enforcement of liquor licence conditions"],
        "sentences": [
            "Licensing inspectors completed {n2} compliance checks on takeaway outlets in {place}.",
            "Two licensees were issued warnings for breaching trading hour conditions.",
            "The liquor licensing compliance program focuses on secondary supply and identification checks.",
            "Inspectors found the bottle shop in {place} was not scanning identification at the point of sale.",
            "Compliance visits will increase during the wet season when trading patterns change.",
            "The licensing team reviewed CCTV retention by licensed venues as part of routine compliance.",
        ],
    },
    "public housing tenancy disputes": {
        "agency": "Housing",
        "phrasings": ["public housing tenancy disputes", "disputes with public housing tenants",
                      "tenancy disputes in public housing"],
        "sentences": [
            "The tenancy team is managing {n2} disputes in {place} public housing.",
            "Most tenancy disputes concern rent arrears, property damage and visitor numbers.",
            "Mediation resolved the dispute before it reached the tribunal.",
            "A notice to remedy breach was issued for repeated noise complaints at the complex.",
            "Tenancy officers attended a hearing at the NT Civil and Administrative Tribunal.",
            "The dispute resolution process was updated to include an interpreter where needed.",
        ],
    },
    "emergency services staffing in remote regions": {
        "agency": "Police",
        "phrasings": ["emergency services staffing in remote regions", "staffing of police and fire stations in remote areas",
                      "remote emergency services workforce levels"],
        "sentences": [
            "The {community} station operated with {n1} officers for most of the quarter.",
            "Relief rosters were used to cover vacancies at remote fire and emergency stations.",
            "Recruitment for remote emergency services positions remains below target.",
            "Fly-in fly-out arrangements are being trialled to staff the {community} station.",
            "Overtime costs for remote emergency services rose because of unfilled positions.",
            "Volunteer emergency service units in {community} received new training equipment.",
        ],
    },
    "regional infrastructure funding allocation": {
        "agency": "Infrastructure",
        "phrasings": ["regional infrastructure funding allocation", "how regional infrastructure funding was allocated",
                      "funding for roads and infrastructure in the regions"],
        "sentences": [
            "The regional infrastructure program allocated funding to {n2} road and bridge projects.",
            "Upgrades to the {community} access road were prioritised because of wet season closures.",
            "Funding for barge landing improvements was split across three coastal communities.",
            "Project scoring for regional infrastructure considers safety, access and cost.",
            "Construction on the {place} causeway is scheduled to start after the wet season.",
            "Regional councils were invited to nominate infrastructure priorities for the next round.",
        ],
    },
    "grant funding for community programs": {
        "agency": "Treasury",
        "phrasings": ["grant funding for community programs", "community program grants",
                      "grants awarded to community organisations"],
        "sentences": [
            "The community grants round received {n2} applications from {place} organisations.",
            "Grant recipients must provide an acquittal report within three months of project completion.",
            "Youth diversion and school holiday programs received the largest share of community grants.",
            "The assessment panel scored applications against the published grant guidelines.",
            "Late applications for the community grants round were not accepted.",
            "Grant agreements for successful community programs were issued in {month}.",
        ],
    },
    "community safety incident reporting": {
        "agency": "Police",
        "phrasings": ["community safety incident reporting", "how community safety incidents are reported",
                      "reports of community safety incidents"],
        "sentences": [
            "Community safety patrols recorded {n2} incidents in {community} during the month.",
            "The new incident reporting form reduced duplicate reports from night patrols.",
            "Incident data is shared monthly with the {community} community safety committee.",
            "Reported incidents fell after lighting was installed near the {community} store.",
            "Community safety officers attended a refresher course on incident reporting.",
            "The quarterly incident report was tabled at the regional safety meeting.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Sensitive phrasings. Each entry: (template, split). split="test" phrasings
# are never used to write rules or train the classifier.
# ---------------------------------------------------------------------------
SENSITIVE = {
    "s56_privacy": [
        ('The tenant {name} at {addr} is behind on rent.', "dev"),
        ('{name}, aged {n2}, has applied for priority housing.', "dev"),
        ("The complainant's partner, {name}, also lives at the property and works night shifts.", "test"),
        ('Send the hardship form to {email} and copy in her sister.', "test"),
        ('{name} told the officer he had been sleeping in his car since losing the lease.', "test"),
        ('Records list {name} as the next of kin for the tenancy.', "test"),
        ("{name} (DOB {dob}) of {addr} lodged a complaint about the handling of their tenancy.", "dev"),
        ("Please call the tenant, {name}, on {mobile} to arrange access.", "dev"),
        ("{name}'s client number is {clientno} and their current address is {addr}.", "dev"),
        ("The applicant, {name}, can be contacted at {email}.", "dev"),
        ("Our records show {name} was evicted from {addr} in {month}.", "dev"),
        ("{name} has asked that their home phone ({landline}) not be shared with the contractor.", "dev"),
        ("Mrs {surname} rang again from {mobile} asking about her rent arrears of ${n3}.", "dev"),
        ("Born on {dob_words}, {name} is the primary leaseholder for the property.", "dev"),
        ("Neighbours named the occupant as {name}, who moved in during {month}.", "dev"),
    ],
    "s56_health": [
        ('{name} has been diagnosed with diabetes and needs regular access to a clinic.', "dev"),
        ('The household includes a person with a psychiatric illness who receives weekly visits.', "dev"),
        ("Surgery is scheduled for the client's hip next month.", "dev"),
        ('He mentioned he is on a waiting list for knee surgery and cannot climb the stairs.', "test"),
        ('{name} has been attending the drug and alcohol counselling service weekly.', "test"),
        ('The occupational therapist recommended grab rails in the bathroom for the tenant.', "test"),
        ('Her psychiatrist has asked that contact be made through the case manager only.', "test"),
        ("{name} is receiving dialysis three times a week at the {place} renal unit.", "dev"),
        ("The tenant's son has a diagnosed disability and uses a wheelchair, so the ramp repair is urgent.", "dev"),
        ("{name} was admitted to Royal Darwin Hospital after the incident.", "dev"),
        ("Medical certificates provided by {name} confirm a mental health condition.", "dev"),
        ("The client is on medication for a heart condition and cannot tolerate the heat without air conditioning.", "dev"),
        ("{name} disclosed a pregnancy when requesting the transfer.", "dev"),
        ("Her GP letter says the mould is aggravating her asthma.", "dev"),
        ("{name} is undergoing chemotherapy and asked for repairs to be scheduled around treatment days.", "dev"),
        ("The family's youngest child has rheumatic heart disease and needs a cool, clean home.", "dev"),
    ],
    "s56_cultural": [
        ('Out of respect, the image of the late leader has been removed from the slides.', "dev"),
        ('Restricted ceremonial knowledge was shared with the anthropologist in confidence.', "dev"),
        ('Only senior women may view the painting, so it must not appear in the published report.', "test"),
        ('The clan requested that recordings of the song cycle remain with the community.', "test"),
        ('Burial grounds near the outstation are not to be shown on any public map.', "test"),
        ("During the mourning period the family's house should not be entered by crews.", "test"),
        ("Contractors must not enter the sacred site behind the {community} airstrip without custodian approval.", "dev"),
        ("Works near the ceremony ground were paused for men's business.", "dev"),
        ("The community asked that the name of the deceased elder not be used during sorry business.", "dev"),
        ("The restricted women's site at {community} is marked on the attached map and must not be published.", "dev"),
        ("Traditional owners shared information about a dreaming track that is restricted to initiated men.", "dev"),
        ("The AAPA certificate notes a sacred site within 200 metres of the proposed works.", "dev"),
        ("Photos of the deceased community member should be removed from the newsletter out of respect.", "dev"),
        ("Senior custodians asked that the location of the ceremonial objects be kept confidential.", "dev"),
        ("Crews will stay off the country east of the creek while initiation ceremonies are under way.", "dev"),
    ],
    "s45_cabinet": [
        ('The Budget Review Subcommittee will consider the bid in March.', "dev"),
        ('Ministers discussed the proposal in Cabinet and asked for further costings.', "dev"),
        ('The paper goes to Cabinet next Tuesday and the recommendations are embargoed until then.', "test"),
        ('Coordination comments from agencies on the draft Cabinet paper are summarised below.', "test"),
        ("The Chief Minister's office has asked that the Cabinet outcome not be shared with stakeholders.", "test"),
        ('This attachment formed part of the Executive Council papers for the {month} meeting.', "test"),
        ("This submission is CABINET-IN-CONFIDENCE and was prepared for Cabinet's consideration.", "dev"),
        ("The draft Cabinet submission recommends a ${n2} million package that has not yet been approved.", "dev"),
        ("Cabinet deliberated on the options at its {month} meeting and has not announced a decision.", "dev"),
        ("Attachment B is a Cabinet decision record and is not for circulation outside the Department.", "dev"),
        ("This briefing reflects draft policy positions prepared for Cabinet and not yet adopted by Government.", "dev"),
        ("Ministers are yet to agree the preferred option set out in the Cabinet paper.", "dev"),
        ("The Executive Council minute was signed on {date_words} but remains confidential.", "dev"),
        ("Please keep the coordination comments on the Budget Review Subcommittee paper within the working group.", "dev"),
        ("The Expenditure Review Committee is still considering the proposal and its contents are confidential.", "dev"),
    ],
    "s46_law_enforcement": [
        ('Police are seeking a warrant to search the vehicle.', "dev"),
        ('The identity of the witness must be protected while charges are pending.', "dev"),
        ('Detectives are interviewing persons of interest and have not laid charges.', "test"),
        ('The name of the person who reported the assault must be withheld for their safety.', "test"),
        ('Patrol vehicles will be stationed near the outlet on Friday nights to catch secondary supply.', "test"),
        ('The lock combination for the evidence room is recorded on page two.', "test"),
        ("An active investigation, reference {inv}, is under way and the suspect has not been notified.", "dev"),
        ("The informant, {name}, provided the tip-off on the condition their identity is protected.", "dev"),
        ("Police intelligence indicates that sly grogging is occurring from {addr}.", "dev"),
        ("Disclosure of the surveillance schedule would prejudice the ongoing operation.", "dev"),
        ("The alarm codes for the {facility} have been changed following the break-in.", "dev"),
        ("CCTV coverage has a blind spot at the rear loading dock of the {facility}.", "dev"),
        ("Officers plan to execute a search warrant at the premises next week.", "dev"),
        ("A confidential source inside the licensee's business reported cash sales after hours.", "dev"),
        ("The covert patrol roster for the long weekend is attached.", "dev"),
    ],
    "s49d_legal_privilege": [
        ('Our lawyers advise that the claim is unlikely to succeed.', "dev"),
        ('The barrister recommends settling the matter before the hearing.', "dev"),
        ('Our in-house legal team considers the contractor would win if the matter went to court.', "test"),
        ('Attached is the draft defence prepared by the Solicitor for the Northern Territory.', "test"),
        ('Advice received from external solicitors on the lease dispute is summarised for the executive.', "test"),
        ('The legal opinion recommends we do not admit fault in correspondence with the tenant.', "test"),
        ("This advice from the Solicitor for the Northern Territory is subject to legal professional privilege.", "dev"),
        ("Legal advice: the Department is likely to be found liable if the repairs remain outstanding.", "dev"),
        ("PRIVILEGED AND CONFIDENTIAL: prepared for the dominant purpose of giving legal advice.", "dev"),
        ("Counsel's opinion on the prospects of the tenancy appeal is attached.", "dev"),
        ("The Solicitor for the Northern Territory advises that the termination clause is unlikely to be enforceable.", "dev"),
        ("This email is legally privileged and should not be forwarded.", "dev"),
        ("Our lawyers consider the claim weak and recommend settling below ${n3},000.", "dev"),
        ("The barrister's assessment is that the licence cancellation may not survive review.", "dev"),
        ("Notes from the conference with Crown counsel on the appeal strategy are attached.", "dev"),
    ],
    "s52_deliberative": [
        ('This is my preliminary view and has not been discussed with the executive.', "dev"),
        ('We are still weighing two options and have not made a recommendation.', "dev"),
        ('Before we settle on a position, can we test the idea of merging the two programs?', "test"),
        ('These are rough working notes and have not been checked by the director.', "test"),
        ('The three options below are still being assessed and no recommendation has been made.', "test"),
        ("I'd lean towards option B but would like the team's views first.", "test"),
        ("These notes are a preliminary, pre-decisional discussion and do not represent the Department's final position.", "dev"),
        ("Option 3 is our preferred approach but has not been endorsed by the executive.", "dev"),
        ("Draft recommendation for discussion only: reduce the program to two communities.", "dev"),
        ("Officers canvassed closing the regional office as one possible option.", "dev"),
        ("This is a working draft and the figures may change before the final report.", "dev"),
        ("For internal consultation: we are weighing whether to cap emergency call-outs.", "dev"),
        ("My personal view is that the trial should not be extended, but this is not yet settled.", "dev"),
        ("Thinking out loud, could we fold the grant round into next year's budget bid?", "dev"),
        ("Early modelling suggests three scenarios and none has been chosen.", "dev"),
    ],
    "s57_commercial": [
        ("The tenderer's financial statements were provided in confidence.", "dev"),
        ("The company's hourly rates are higher than the market average.", "dev"),
        ('{company} asked that its labour hire rates be kept confidential under the tender conditions.', "test"),
        ("The bid included a breakdown of the company's overheads and subcontractor costs.", "test"),
        ('Their revenue figures from the last three years were supplied with the tender response.', "test"),
        ("{company}'s quote was ${n2},000 below the next lowest, using its own imported panels.", "test"),
        ("{company}'s tender pricing is commercial-in-confidence and must not be released to competitors.", "dev"),
        ("The schedule of rates shows an emergency call-out unit rate of ${n3}.", "dev"),
        ("{company} submitted a pricing breakdown of ${n3},000 for labour and materials.", "dev"),
        ("The contractor's profit margin of {n1}{n1} per cent was disclosed in confidence during negotiation.", "dev"),
        ("Evaluation scores for each tenderer are attached and {company} scored {n2} out of 100.", "dev"),
        ("The supplier's proprietary maintenance scheduling method is a trade secret.", "dev"),
        ("{company} offered a 12 per cent discount if awarded both regions, an offer made in confidence.", "dev"),
        ("Their insurance premiums and staff wage costs are set out in the bid financials.", "dev"),
        ("The unsuccessful bidder's hourly labour charge was ${n2} lower than the winner's.", "dev"),
    ],
}

# Look-alike sentences that are NOT sensitive. They test over-flagging.
DECOYS = [
    "The Cabinet-approved funding was announced in the {year} Budget and is publicly available.",
    "The Minister's media release about the program is attached for information.",
    "Please contact the Department's FOI unit on 08 8999 5511 with any questions.",
    "The total contract value of ${n2} million was published on the Quotations and Tenders Online website.",
    "Police issued a public road safety warning ahead of the long weekend.",
    "The kitchen cabinet doors in unit 4 have been replaced.",
    "Staff completed mandatory first aid and workplace health and safety training.",
    "The Land Council's annual report is available on its website.",
    "The final policy was endorsed by the executive and published in {month}.",
    "General information about tenancy rights is on the NT Consumer Affairs website.",
    "The chief executive, {official}, signed the published annual report.",
    "Hospital car parking changes were announced on the Department's website.",
]

OPENERS = {
    "Email": ["Hi {first},", "Good morning {first},", "{first},"],
    "Briefing note": ["PURPOSE: To brief the Minister on {topic}.", "MINISTERIAL BRIEF: {topic}."],
    "Memo": ["MEMO: {topic}.", "Internal memo regarding {topic}."],
    "Meeting minutes": ["Minutes of the {topic} working group.", "Meeting notes: {topic}."],
    "Report": ["Summary report on {topic}.", "Quarterly report: {topic}."],
}
CLOSERS = {
    "Email": ["Regards, {author_first}", "Thanks, {author_first}"],
    "Briefing note": ["Recommendation: that the Minister note this brief.", "Contact officer: {author}."],
    "Memo": ["Please action by close of business Friday.", "For your information."],
    "Meeting minutes": ["Next meeting to be scheduled.", "Actions were assigned to members."],
    "Report": ["This report will be updated next quarter.", "End of report."],
}

OFFICIALS = ["Marion Teale", "Graham Okafor", "Priya Natarajan"]

# Subject lines officers actually write: short and inconsistent, rarely the
# formal topic name.
SUBJECTS = {
    "remote housing maintenance backlog": ["Repairs update {community}", "Outstanding maintenance jobs", "Trades hub trial"],
    "alcohol management plan review": ["AMP review {community}", "Alcohol plan consultation", "AMP terms of reference"],
    "liquor licensing compliance": ["Licensee compliance checks", "Bottle shop inspections {place}", "Liquor compliance"],
    "public housing tenancy disputes": ["Tenancy matter {place}", "NTCAT hearing", "Dispute update"],
    "emergency services staffing in remote regions": ["{community} station roster", "Remote vacancies", "Relief staffing"],
    "regional infrastructure funding allocation": ["Regional roads program", "{community} access road", "Infrastructure round"],
    "grant funding for community programs": ["Community grants round", "Grant acquittals", "Grants assessment"],
    "community safety incident reporting": ["Night patrol incidents {community}", "Incident reporting form", "Safety committee data"],
}


def _fill(template, rng, ctx):
    """Fill a template with fresh fake values. Values are generated per call so
    names, numbers and addresses vary across records."""
    vals = dict(ctx)
    name = fake.name()
    vals.update(
        name=name,
        surname=name.split()[-1],
        dob=fake.date_of_birth(minimum_age=19, maximum_age=85).strftime("%d/%m/%Y"),
        dob_words=fake.date_of_birth(minimum_age=19, maximum_age=85).strftime("%-d %B %Y"),
        addr=f"{fake.building_number()} {fake.street_name()} {fake.street_suffix()}, {rng.choice(NT_PLACES)}",
        mobile=f"04{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)}",
        landline=f"08 89{rng.randint(10, 99)} {rng.randint(1000, 9999)}",
        email=f"{name.split()[0].lower()}.{name.split()[-1].lower()}@example.com",
        clientno=f"HC-{rng.randint(100000, 999999)}",
        inv=f"INV-{rng.randint(2023, 2026)}-{rng.choice('ABCDEFGH')}{rng.choice('KLMNPRST')}{rng.randint(10, 99)}",
        company=fake.company(),
        place=rng.choice(NT_PLACES),
        community=rng.choice(COMMUNITIES),
        facility=rng.choice(FACILITIES),
        month=rng.choice(["January", "March", "May", "July", "September", "November"]),
        year=rng.choice(["2024-25", "2025-26"]),
        date_words=f"{rng.randint(1, 28)} {rng.choice(['February', 'April', 'June', 'August'])} {rng.choice([2024, 2025])}",
        official=rng.choice(OFFICIALS),
        n1=str(rng.randint(2, 9)),
        n2=str(rng.randint(11, 98)),
        n3=str(rng.randint(120, 980)),
    )
    return template.format(**vals)


def generate(n_records=500, n_requests=40, sensitive_rate=0.42, out_dir=None):
    rng = random.Random(SEED)
    Faker.seed(SEED)
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    topic_names = list(TOPICS)
    start, span_days = date(2023, 1, 1), (date(2026, 9, 30) - date(2023, 1, 1)).days
    records, gt_records = [], []

    for i in range(1, n_records + 1):
        topic = rng.choice(topic_names)
        t = TOPICS[topic]
        agency_key = t["agency"] if rng.random() < 0.8 else rng.choice(list(AGENCIES))
        cc = []
        if rng.random() < 0.15:
            cc = [AGENCIES[rng.choice([k for k in AGENCIES if k != agency_key])]]
        dtype = rng.choice(DOC_TYPES)
        d = start + timedelta(days=rng.randint(0, span_days))
        author, recipient = fake.name(), fake.name()
        ctx = {"topic": topic, "first": recipient.split()[0], "author": author, "author_first": author.split()[0]}

        content = [_fill(s, rng, ctx) for s in rng.sample(t["sentences"], k=rng.randint(3, 4))]
        labels = []  # (category, split, is_decoy) per inserted sentence, matched later to offsets

        is_sensitive = rng.random() < sensitive_rate
        split = "test" if rng.random() < 0.3 else "dev"
        inserted = []
        if is_sensitive:
            n_cats = 2 if rng.random() < 0.2 else 1
            for cat in rng.sample(list(SENSITIVE), k=n_cats):
                pool = [p for p in SENSITIVE[cat] if p[1] == split]
                tmpl, _ = rng.choice(pool)
                inserted.append((_fill(tmpl, rng, ctx), cat))
        if rng.random() < 0.22:
            inserted.append((_fill(rng.choice(DECOYS), rng, ctx), None))
        for sent, cat in inserted:
            content.insert(rng.randint(1, len(content)), sent)

        opener = _fill(rng.choice(OPENERS[dtype]), rng, ctx)
        closer = _fill(rng.choice(CLOSERS[dtype]), rng, ctx)
        sentences = [opener] + content + [closer]
        if cc:
            sentences.insert(1, f"Copied to {cc[0]} for information.")
        body = " ".join(sentences)

        spans = []
        for sent, cat in inserted:
            s = body.index(sent)
            spans.append({"start": s, "end": s + len(sent), "text": sent,
                          "category": cat, "decoy": cat is None})

        rid = f"REC-{i:05d}"
        records.append({
            "record_id": rid,
            "agency": AGENCIES[agency_key],
            "cc": cc,
            "doc_type": dtype,
            "date": d.isoformat(),
            "from": author,
            "to": recipient,
            "subject": _fill(rng.choice(SUBJECTS[topic]), rng, ctx),
            "body": body,
        })
        gt_records.append({
            "record_id": rid,
            "topic": topic,
            "split": split,
            "sensitive": any(not sp["decoy"] for sp in spans),
            "categories": sorted({sp["category"] for sp in spans if not sp["decoy"]}),
            "spans": spans,
        })

    # -------------------------- requests --------------------------
    templates = [
        ("All documents and correspondence about {phr} held by {agency}, dated {d1} to {d2}.", {"agency", "dates"}, None),
        ("Any briefing notes prepared for the Minister about {phr} between {d1} and {d2}.", {"dates"}, "Briefing note"),
        ("Copies of internal memos discussing {phr} since {d1}.", {"open"}, "Memo"),
        ("Emails between {agency} and contractors regarding {phr} from {d1} onward.", {"agency", "open"}, "Email"),
        ("Records about {phr} from {m1} to {m2}.", {"months"}, None),
    ]
    requests, gt_requests = [], []
    for j in range(1, n_requests + 1):
        topic = topic_names[(j - 1) % len(topic_names)]
        tmpl, flags, dtype = templates[(j - 1) % len(templates)]
        phr = rng.choice(TOPICS[topic]["phrasings"])
        agency_key = TOPICS[topic]["agency"] if rng.random() < 0.75 else rng.choice(list(AGENCIES))
        d1 = start + timedelta(days=rng.randint(0, span_days - 400))
        d2 = d1 + timedelta(days=rng.randint(240, 700))
        if d2 > date(2026, 9, 30):
            d2 = date(2026, 9, 30)
        if "open" in flags:
            d2 = date(2026, 9, 30)
        if "months" in flags:
            d1 = d1.replace(day=1)
            nxt = (d2.replace(day=28) + timedelta(days=4))
            d2 = nxt - timedelta(days=nxt.day)  # end of month
        text = tmpl.format(phr=phr, agency=AGENCIES[agency_key],
                           d1=d1.strftime("%-d %B %Y"), d2=d2.strftime("%-d %B %Y"),
                           m1=d1.strftime("%B %Y"), m2=d2.strftime("%B %Y"))
        use_agency = "agency" in flags
        responsive = []
        for r, g in zip(records, gt_records):
            if g["topic"] != topic:
                continue
            if not (d1.isoformat() <= r["date"] <= d2.isoformat()):
                continue
            if use_agency and not (r["agency"] == AGENCIES[agency_key] or AGENCIES[agency_key] in r["cc"]):
                continue
            if dtype and r["doc_type"] != dtype:
                continue
            responsive.append(r["record_id"])
        rq = f"REQ-{j:04d}"
        requests.append({"request_id": rq, "request_text": text, "received": "2026-09-21"})
        gt_requests.append({"request_id": rq, "topic": topic, "agency": AGENCIES[agency_key] if use_agency else None,
                            "date_from": d1.isoformat(), "date_to": d2.isoformat(), "doc_type": dtype,
                            "responsive_record_ids": responsive})

    def dump(name, rows):
        with open(out_dir / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    dump("records.jsonl", records)
    dump("requests.jsonl", requests)
    dump("ground_truth_records.jsonl", gt_records)
    dump("ground_truth_requests.jsonl", gt_requests)
    return records, requests, gt_records, gt_requests


if __name__ == "__main__":
    recs, reqs, gtr, gtq = generate()
    n_s = sum(g["sensitive"] for g in gtr)
    print(f"{len(recs)} records ({n_s} sensitive), {len(reqs)} requests")
