"""Exemption categories, mapped to the Information Act 2002 (NT).

Section references are to the Act as consolidated on legislation.nt.gov.au:
  s45  Executive Council, Cabinet, Territory economy        (Part 4 Div 2)
  s46  Security and law enforcement                          (Part 4 Div 2)
  s49  Preservation of system of justice; s49(d) covers client legal privilege
  s52  Deliberative processes                                (Part 4 Div 3, public interest test)
  s56  Privacy and cultural information                      (Part 4 Div 3, public interest test)
  s57  Commercial and business information                   (Part 4 Div 3, public interest test)

Division 3 exemptions only apply if disclosure is shown to be against the
public interest (s50). That weighing is a human judgement: the system flags
candidate material, it never decides that an exemption applies.
"""

CATEGORIES = {
    "s56_privacy": {
        "label": "Personal information",
        "section": "s56 Privacy and cultural information",
        "high_harm": False,
    },
    "s56_health": {
        "label": "Health information about a person",
        "section": "s56 Privacy and cultural information",
        "high_harm": True,
    },
    "s56_cultural": {
        "label": "Aboriginal sacred or restricted information",
        "section": "s56 Privacy and cultural information (see also s54 places of significance)",
        "high_harm": True,
    },
    "s45_cabinet": {
        "label": "Cabinet or Executive Council material",
        "section": "s45 Executive Council, Cabinet, Territory economy",
        "high_harm": False,
    },
    "s46_law_enforcement": {
        "label": "Security or law enforcement",
        "section": "s46 Security and law enforcement",
        "high_harm": True,
    },
    "s49d_legal_privilege": {
        "label": "Legal advice (privilege)",
        "section": "s49(d) Client legal privilege",
        "high_harm": False,
    },
    "s52_deliberative": {
        "label": "Deliberative material",
        "section": "s52 Deliberative processes",
        "high_harm": False,
    },
    "s57_commercial": {
        "label": "Commercial or business information",
        "section": "s57 Commercial and business information",
        "high_harm": False,
    },
}

CATEGORY_IDS = list(CATEGORIES)

# High-harm categories: the damage from a wrong release is severe and hard to
# undo (health details, sacred knowledge, informant safety). These are routed
# to a human at a lower confidence floor than everything else.
HIGH_HARM = {c for c, m in CATEGORIES.items() if m["high_harm"]}
