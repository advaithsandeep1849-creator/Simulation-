#!/usr/bin/env python3
"""
TTP Impact Simulator — local server (v2)

Run:
    pip install flask numpy --break-system-packages
    python3 app.py
Then open http://localhost:5000

v2 adds: parameter provenance/evidence tracking, seeded reproducible Monte
Carlo, a validation/sanity-check layer, configurable recovery/recycling
rates (no hard-coded multipliers), an expanded verification funnel
(evidence, package matching, appeals), bounded per-agent behavioural
response to violations, a location/environment breakdown, sensitivity
analysis, and scenario comparison. All existing routes and behaviour from
v1 are preserved; nothing was removed.
"""

import json
import urllib.request
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

DEFAULT_PARAMS = {
    "deterrenceEffect": 0.55,
    "adoptionGrowthRate": 0.18,
    "reportDetectionRate": 0.40,
    "fineAccuracy": 0.85,          # semantically: verification pass rate
    "hksBaseCollectionRate": 0.15,
}

# Neutral (1.0) multipliers by default — deliberately NOT differentiated,
# since no real per-location Kerala data has been supplied. Any deviation
# from 1.0 is the user's own explicit assumption, entered and labelled
# through the Baseline & Evidence panel.
DEFAULT_LOCATIONS = [
    {"name": "Residential", "populationShare": 0.50},
    {"name": "School", "populationShare": 0.10},
    {"name": "College/Campus", "populationShare": 0.10},
    {"name": "Market", "populationShare": 0.10},
    {"name": "Roadside/Public street", "populationShare": 0.10},
    {"name": "Beach/Coastal", "populationShare": 0.05},
    {"name": "Tourist area", "populationShare": 0.05},
]
for _loc in DEFAULT_LOCATIONS:
    _loc.update({
        "baseLitterMultiplier": 1.0, "perCapitaMultiplier": 1.0,
        "adoptionMultiplier": 1.0, "reportingMultiplier": 1.0,
        "collectionMultiplier": 1.0,
    })

CALIBRATION_PROMPT = """You are calibrating a behavioural simulation of the Trash Track Programme (TTP), \
a QR-code-based plastic accountability system in Kerala, India. Mechanism: every plastic product has a \
persistent QR identity linked to a citizen's TTP ID. Littered items can be scanned and reported, triggering \
a fine to the responsible TTP ID after verification. Properly disposed items can be collected by an authorised \
Haritha Karma Sena (HKS) unit. Ownership can be legitimately transferred (gifting) so responsibility follows \
the current owner. Based on how deterrence-plus-incentive civic programmes typically diffuse through a \
population, return ONLY a JSON object with these numeric fields (plain decimals between 0 and 1): \
deterrenceEffect, adoptionGrowthRate, reportDetectionRate, fineAccuracy, hksBaseCollectionRate, and a short \
string field "reasoning" (2-3 sentences explaining your choices)."""


# ---------------------------------------------------------------------------
# AI calibration proxy — Groq (OpenAI-compatible, free tier, no card
# required). Timestamps calls for provenance so AI-derived numbers can be
# distinguished from evidence.
# ---------------------------------------------------------------------------

def call_ai(api_key, prompt, json_mode=False):
    body = {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": prompt}], "temperature": 0.4}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (compatible; TTP-Impact-Simulator/1.0)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    import datetime
    payload = request.get_json(force=True)
    api_key = payload.get("apiKey", "").strip()
    if not api_key:
        return jsonify({"error": "No API key provided"}), 400
    try:
        content = call_ai(api_key, CALIBRATION_PROMPT, json_mode=True)
        parsed = json.loads(content)
        params = {**DEFAULT_PARAMS, **parsed}
        params["_provenance"] = {
            "sourceType": "AI-assisted calibration / model assumption",
            "generatedAt": datetime.datetime.utcnow().isoformat() + "Z",
            "calibrationPrompt": CALIBRATION_PROMPT,
        }
        return jsonify(params)
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/narrative", methods=["POST"])
def api_narrative():
    payload = request.get_json(force=True)
    api_key = payload.get("apiKey", "").strip()
    s = payload.get("summary", {})
    cfg = payload.get("cfg", {})
    if not api_key:
        return jsonify({"error": "No API key provided"}), 400
    prompt = (
        f"Here is the output of a Monte Carlo simulation ({cfg.get('numRuns')} runs) of the Trash Track "
        f"Programme (TTP), a QR-based plastic accountability system: median litter-rate reduction "
        f"{s['litterDropPct']['med']:.1f}% (range {s['litterDropPct']['lo']:.1f}-{s['litterDropPct']['hi']:.1f}%), "
        f"median adoption reached {s['finalAdoptionPct']['med']:.0f}%, median "
        f"{round(s['totalRecovered']['med']):,} items recovered, median {round(s['totalVerified']['med']):,} "
        f"verified violations after filtering {round(s['totalDuplicates']['med']):,} duplicates and "
        f"{round(s['totalFalseFlagged']['med']):,} false reports. These are simulation projections under "
        f"stated assumptions, not measured real-world data. Write a short (3 short paragraphs, no headers, "
        f"no bullet points) plain-language interpretation of what this range of outcomes would mean for a "
        f"community if TTP were deployed at this scale — covering behavioural change, accountability/reporting "
        f"integrity, and one honest caveat about the limits of this projection. Write in a grounded, non-hype "
        f"tone suitable for a project report. Do not claim this is observed or measured data."
    )
    try:
        text = call_ai(api_key, prompt, json_mode=False)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def normalize_cfg(cfg):
    """Clamp/patch inputs and fill defaults for any new v2 fields the
    frontend didn't send, so the engine never receives missing/invalid
    values silently."""
    cfg = dict(cfg)
    cfg["months"] = max(1, min(120, int(cfg.get("months", 24))))
    cfg["numRuns"] = max(5, min(60, int(cfg.get("numRuns", 30))))
    cfg["agentSample"] = max(50, min(5000, int(cfg.get("agentSample", 3000))))
    cfg["heterogeneity"] = max(0.0, min(0.9, float(cfg.get("heterogeneity", 0.4))))
    cfg["jitter"] = max(0.0, min(0.8, float(cfg.get("jitter", 0.2))))
    cfg["adoptionCeiling"] = max(0.0, min(1.0, float(cfg.get("adoptionCeiling", 0.5))))
    cfg["population"] = max(1, int(cfg.get("population", 200000)))
    cfg["perCapita"] = max(0.0, float(cfg.get("perCapita", 6)))
    cfg["baseLitter"] = max(0.0, min(1.0, float(cfg.get("baseLitter", 0.35))))
    cfg["fineBase"] = max(0.0, float(cfg.get("fineBase", 100)))
    cfg["fineStep"] = max(0.0, float(cfg.get("fineStep", 100)))
    cfg["fineCap"] = max(cfg["fineBase"], float(cfg.get("fineCap", 500)))
    cfg["hksGrowth"] = max(0.0, min(1.0, float(cfg.get("hksGrowth", 0.04))))
    cfg["transferRate"] = max(0.0, min(1.0, float(cfg.get("transferRate", 0.05))))
    cfg["biometricCeiling"] = max(0.0, min(1.0, float(cfg.get("biometricCeiling", 0.6))))
    cfg["qrFailureRate"] = max(0.0, min(0.95, float(cfg.get("qrFailureRate", 0.08))))
    cfg["falseReportRate"] = max(0.0, min(0.95, float(cfg.get("falseReportRate", 0.06))))
    cfg["duplicateRate"] = max(0.0, min(0.95, float(cfg.get("duplicateRate", 0.08))))
    # v2 fields (new configurable rates — replace old hard-coded 0.6 / 0.7)
    cfg["litterRecoveryRate"] = max(0.0, min(1.0, float(cfg.get("litterRecoveryRate", 0.6))))
    cfg["recyclingRate"] = max(0.0, min(1.0, float(cfg.get("recyclingRate", 0.7))))
    cfg["evidenceSubmissionRate"] = max(0.0, min(1.0, float(cfg.get("evidenceSubmissionRate", 0.5))))
    cfg["packageMatchingRate"] = max(0.0, min(1.0, float(cfg.get("packageMatchingRate", 0.9))))
    cfg["appealRate"] = max(0.0, min(1.0, float(cfg.get("appealRate", 0.05))))
    cfg["appealSuccessRate"] = max(0.0, min(1.0, float(cfg.get("appealSuccessRate", 0.3))))
    cfg["behavioralResponseStrength"] = max(0.0, min(0.9, float(cfg.get("behavioralResponseStrength", 0.15))))
    cfg["transferFailureRate"] = max(0.0, min(1.0, float(cfg.get("transferFailureRate", 0.05))))
    cfg["biometricAuthSuccessRate"] = max(0.0, min(1.0, float(cfg.get("biometricAuthSuccessRate", 0.97))))
    seed = cfg.get("seed", None)
    cfg["seed"] = int(seed) if (seed is not None and str(seed) != "") else None
    cfg["locations"] = cfg.get("locations") or DEFAULT_LOCATIONS
    return cfg


def run_simulation(cfg, base_params):
    import numpy as np

    cfg = normalize_cfg(cfg)
    rng = np.random.default_rng(cfg["seed"])  # None -> non-deterministic, as before

    R = cfg["numRuns"]
    N = cfg["agentSample"]
    months = cfg["months"]
    scale = cfg["population"] / N
    spread = cfg["heterogeneity"]
    qr_reliability = 1 - cfg["qrFailureRate"]
    pct = cfg["jitter"]

    def jitter(v, lo, hi):
        factor = 1 + (rng.random(R) * 2 - 1) * pct
        return np.clip(v * factor, lo, hi)

    deterrenceEffect = jitter(base_params.get("deterrenceEffect", 0.55), 0.1, 0.95)
    adoptionGrowthRate = jitter(base_params.get("adoptionGrowthRate", 0.18), 0.03, 0.6)
    reportDetectionRate = jitter(base_params.get("reportDetectionRate", 0.40), 0.05, 0.95)
    verificationPassRate = jitter(base_params.get("fineAccuracy", 0.85), 0.3, 0.99)
    hksRate = jitter(base_params.get("hksBaseCollectionRate", 0.15), 0.02, 0.6)

    # --- location assignment (fixed per agent for the whole run) ---
    locs = cfg["locations"]
    L = len(locs)
    shares = np.array([max(0.0001, l.get("populationShare", 1.0 / L)) for l in locs])
    shares = shares / shares.sum()
    loc_index = rng.choice(L, size=(R, N), p=shares)
    base_litter_mult_vec = np.array([l.get("baseLitterMultiplier", 1.0) for l in locs])
    per_capita_mult_vec = np.array([l.get("perCapitaMultiplier", 1.0) for l in locs])
    adoption_mult_vec = np.array([l.get("adoptionMultiplier", 1.0) for l in locs])
    reporting_mult_vec = np.array([l.get("reportingMultiplier", 1.0) for l in locs])
    collection_mult_vec = np.array([l.get("collectionMultiplier", 1.0) for l in locs])
    baseLitterMult = base_litter_mult_vec[loc_index]
    perCapitaMult = per_capita_mult_vec[loc_index]
    adoptionMult = adoption_mult_vec[loc_index]
    reportingMult = reporting_mult_vec[loc_index]
    collectionMult = collection_mult_vec[loc_index]

    tendency0 = np.maximum(0.05, 1 - spread + rng.random((R, N)) * 2 * spread)
    tendency = tendency0.copy()  # mutable — bounded behavioural response reduces this over time
    enrollRank = rng.random((R, N))
    biometricRank = rng.random((R, N))
    violationCount = np.zeros((R, N))

    adoptionFrac = np.full(R, 0.02 if cfg["adoptionCeiling"] > 0 else 0.0)
    biometricFrac = np.full(R, 0.02)
    cumulativeCollected = np.zeros(R)
    cumulativeRecycled = np.zeros(R)
    cumulativeAuthorities = np.zeros(R)
    cumulativeTransferredFailed = np.zeros(R)

    fields = ["adoption", "biometric", "litterRate", "properRate", "reportsRaw", "duplicates",
              "falseFlagged", "withEvidence", "matched", "verified", "confirmed", "rejected",
              "appeals", "successfulAppeals", "unsuccessfulAppeals",
              "fineRevenue", "collectedThisMonth", "cumulativeCollected",
              "recycledThisMonth", "cumulativeRecycled", "transferredItems", "transferredFailed",
              "qrReliability", "qrAttemptedScans", "qrSuccessfulScans", "qrFailedScans",
              "biometricAttempts", "biometricSuccesses", "biometricFailures",
              "authoritiesReferrals", "cumulativeAuthorities"]
    store = {f: np.zeros((months + 1, R)) for f in fields}

    # per-location cumulative accumulators (final-state comparison, section 3)
    loc_cum = {k: np.zeros((L, R)) for k in ["reportsRaw", "verified", "collectedThisMonth"]}

    for m in range(months + 1):
        enrolled = enrollRank < np.clip(adoptionFrac[:, None] * adoptionMult, 0, 1)
        enrolledCount = enrolled.sum(axis=1)
        biometricEnrolled = enrolled & (biometricRank < biometricFrac[:, None])
        biometricCount = biometricEnrolled.sum(axis=1)

        litterRate_i = cfg["baseLitter"] * baseLitterMult * tendency
        litterRate_i = np.where(enrolled, litterRate_i * (1 - deterrenceEffect[:, None]), litterRate_i)
        litterRate_i = np.clip(litterRate_i, 0.02, 0.95)

        perCapita_i = cfg["perCapita"] * perCapitaMult
        litteredItems = perCapita_i * litterRate_i
        properItems = perCapita_i - litteredItems
        litteredTotal = litteredItems.sum(axis=1)
        properTotal = properItems.sum(axis=1)

        readable = litteredItems * qr_reliability
        reportsRaw_i = np.where(enrolled, readable * np.clip(reportDetectionRate[:, None] * reportingMult, 0, 0.98), 0)
        afterDup_i = reportsRaw_i * (1 - cfg["duplicateRate"])
        afterFalse_i = afterDup_i * (1 - cfg["falseReportRate"])
        withEvidence_i = afterFalse_i * cfg["evidenceSubmissionRate"]
        matched_i = withEvidence_i * cfg["packageMatchingRate"]
        verified_prob_i = np.clip(matched_i * verificationPassRate[:, None], 0, 1)

        transferredSum = np.where(enrolled, perCapita_i * cfg["transferRate"], 0).sum(axis=1)

        # Per-agent Bernoulli draw: does THIS agent incur a verified
        # violation this month? Drives persistent violation history and
        # fine escalation (an expected-value probability wouldn't let
        # violations attach to specific repeat-offending individuals).
        draws = rng.random((R, N)) < verified_prob_i
        appeals = draws & (rng.random((R, N)) < cfg["appealRate"])
        appealSuccess = appeals & (rng.random((R, N)) < cfg["appealSuccessRate"])
        confirmed = draws & ~appealSuccess  # stands after any appeal

        fineAmounts = np.minimum(cfg["fineBase"] + violationCount * cfg["fineStep"], cfg["fineCap"])
        referred = violationCount >= 5
        fineRevenueSum = np.where(confirmed, fineAmounts, 0).sum(axis=1)
        referralsSum = (confirmed & referred).sum(axis=1)

        # Bounded behavioural response: a confirmed violation nudges that
        # agent's littering tendency down, with a floor so no one becomes
        # perfectly responsible overnight.
        tendency = np.where(confirmed, np.maximum(0.05, tendency * (1 - cfg["behavioralResponseStrength"])), tendency)
        violationCount = violationCount + confirmed.astype(float)

        littered = litteredTotal * scale
        proper = properTotal * scale
        items = littered + proper
        litterRateAgg = np.where(items > 0, littered / items, 0)
        properRateAgg = 1 - litterRateAgg
        fineRevenue = fineRevenueSum * scale
        authoritiesReferrals = referralsSum * scale
        cumulativeAuthorities = cumulativeAuthorities + authoritiesReferrals

        collectedFromProper_i = properItems * np.clip(hksRate[:, None] * collectionMult, 0, 1)
        collectedFromProperSum = collectedFromProper_i.sum(axis=1) * scale
        collectedFromConfirmedSum = confirmed.sum(axis=1) * scale * cfg["litterRecoveryRate"]
        collectedThisMonth = collectedFromProperSum + collectedFromConfirmedSum
        # cannot recover more than was generated this month
        collectedThisMonth = np.minimum(collectedThisMonth, items)
        cumulativeCollected = cumulativeCollected + collectedThisMonth
        recycledThisMonth = collectedThisMonth * cfg["recyclingRate"]
        cumulativeRecycled = cumulativeRecycled + recycledThisMonth

        transferredFailed = transferredSum * cfg["transferFailureRate"]
        cumulativeTransferredFailed = cumulativeTransferredFailed + transferredFailed

        biometricAttempts = biometricCount * scale
        biometricSuccesses = biometricAttempts * cfg["biometricAuthSuccessRate"]
        biometricFailures = biometricAttempts - biometricSuccesses

        store["adoption"][m] = enrolledCount / N
        store["biometric"][m] = biometricCount / N
        store["litterRate"][m] = litterRateAgg
        store["properRate"][m] = properRateAgg
        store["reportsRaw"][m] = reportsRaw_i.sum(axis=1) * scale
        store["duplicates"][m] = (reportsRaw_i * cfg["duplicateRate"]).sum(axis=1) * scale
        store["falseFlagged"][m] = (afterDup_i * cfg["falseReportRate"]).sum(axis=1) * scale
        store["withEvidence"][m] = withEvidence_i.sum(axis=1) * scale
        store["matched"][m] = matched_i.sum(axis=1) * scale
        store["verified"][m] = draws.sum(axis=1) * scale
        store["confirmed"][m] = confirmed.sum(axis=1) * scale
        store["rejected"][m] = np.maximum(0, (matched_i.sum(axis=1) - matched_i.sum(axis=1) * verificationPassRate) ) * scale
        store["appeals"][m] = appeals.sum(axis=1) * scale
        store["successfulAppeals"][m] = appealSuccess.sum(axis=1) * scale
        store["unsuccessfulAppeals"][m] = (appeals & ~appealSuccess).sum(axis=1) * scale
        store["fineRevenue"][m] = fineRevenue
        store["collectedThisMonth"][m] = collectedThisMonth
        store["cumulativeCollected"][m] = cumulativeCollected
        store["recycledThisMonth"][m] = recycledThisMonth
        store["cumulativeRecycled"][m] = cumulativeRecycled
        store["transferredItems"][m] = transferredSum * scale
        store["transferredFailed"][m] = transferredFailed
        store["qrReliability"][m] = qr_reliability
        store["qrAttemptedScans"][m] = littered
        store["qrSuccessfulScans"][m] = littered * qr_reliability
        store["qrFailedScans"][m] = littered * (1 - qr_reliability)
        store["biometricAttempts"][m] = biometricAttempts
        store["biometricSuccesses"][m] = biometricSuccesses
        store["biometricFailures"][m] = biometricFailures
        store["authoritiesReferrals"][m] = authoritiesReferrals
        store["cumulativeAuthorities"][m] = cumulativeAuthorities

        for l in range(L):
            mask = loc_index == l
            loc_cum["reportsRaw"][l] += np.where(mask, reportsRaw_i, 0).sum(axis=1) * scale
            loc_cum["verified"][l] += np.where(mask, draws, 0).sum(axis=1) * scale
            loc_cum["collectedThisMonth"][l] += np.where(mask, collectedFromProper_i, 0).sum(axis=1) * scale

        if cfg["adoptionCeiling"] > 0:
            adoptionFrac = adoptionFrac + adoptionGrowthRate * adoptionFrac * (cfg["adoptionCeiling"] - adoptionFrac) + 0.0005
            adoptionFrac = np.clip(adoptionFrac, 0, cfg["adoptionCeiling"])
        biometricFrac = np.clip(biometricFrac + 0.15 * biometricFrac * (cfg["biometricCeiling"] - biometricFrac) + 0.0005, 0, cfg["biometricCeiling"])
        hksRate = np.minimum(0.95, hksRate + cfg["hksGrowth"] * hksRate * (1 - hksRate) + cfg["hksGrowth"] * 0.02)

    # final-state per-agent snapshot (last loop iteration's arrays) used for
    # location breakdown and repeat-offender-reduction metric
    everOffended = violationCount > 0
    meaningfullyReduced = tendency < tendency0 * 0.95
    repeatOffenderReduction = np.where(
        everOffended.sum(axis=1) > 0,
        (everOffended & meaningfullyReduced).sum(axis=1) / np.maximum(1, everOffended.sum(axis=1)),
        0,
    ) * 100

    def med_lo_hi(arr_2d):
        return {"med": np.median(arr_2d, axis=1).tolist(), "lo": np.percentile(arr_2d, 10, axis=1).tolist(),
                "hi": np.percentile(arr_2d, 90, axis=1).tolist()}

    per_month = []
    ml = {f: med_lo_hi(store[f]) for f in fields}
    for m in range(months + 1):
        rec = {"m": m}
        for f in fields:
            rec[f] = {"med": ml[f]["med"][m], "lo": ml[f]["lo"][m], "hi": ml[f]["hi"][m]}
        per_month.append(rec)

    def summarize(arr_1d):
        return {"med": float(np.median(arr_1d)), "lo": float(np.percentile(arr_1d, 10)), "hi": float(np.percentile(arr_1d, 90))}

    litter_first = store["litterRate"][0]
    litter_last = store["litterRate"][-1]
    litter_drop_pct = np.where(litter_first > 0, (litter_first - litter_last) / litter_first * 100, 0)

    summary = {
        "litterDropPct": summarize(litter_drop_pct),
        "finalAdoptionPct": summarize(store["adoption"][-1] * 100),
        "finalBiometricPct": summarize(store["biometric"][-1] * 100),
        "totalRecovered": summarize(store["cumulativeCollected"][-1]),
        "totalRecycled": summarize(store["cumulativeRecycled"][-1]),
        "totalFineRevenue": summarize(store["fineRevenue"].sum(axis=0)),
        "totalVerified": summarize(store["verified"].sum(axis=0)),
        "totalConfirmed": summarize(store["confirmed"].sum(axis=0)),
        "totalReportsRaw": summarize(store["reportsRaw"].sum(axis=0)),
        "totalDuplicates": summarize(store["duplicates"].sum(axis=0)),
        "totalFalseFlagged": summarize(store["falseFlagged"].sum(axis=0)),
        "totalAppeals": summarize(store["appeals"].sum(axis=0)),
        "totalSuccessfulAppeals": summarize(store["successfulAppeals"].sum(axis=0)),
        "avgQrReliabilityPct": summarize(store["qrReliability"][0] * 100),
        "totalTransferred": summarize(store["transferredItems"].sum(axis=0)),
        "totalTransferredFailed": summarize(store["transferredFailed"].sum(axis=0)),
        "totalAuthoritiesReferrals": summarize(store["cumulativeAuthorities"][-1]),
        "unreachedPopulation": summarize(np.full(R, cfg["population"] * (1 - cfg["adoptionCeiling"]))),
        "repeatOffenderReductionPct": summarize(repeatOffenderReduction),
        "totalBiometricAttempts": summarize(store["biometricAttempts"].sum(axis=0)),
        "totalBiometricSuccesses": summarize(store["biometricSuccesses"].sum(axis=0)),
    }

    best_idx = int(np.argmin(np.abs(litter_drop_pct - summary["litterDropPct"]["med"])))
    median_run = []
    for m in range(months + 1):
        row = {f: float(store[f][m, best_idx]) for f in fields}
        row["m"] = m
        median_run.append(row)

    location_results = []
    for l, loc in enumerate(locs):
        pop_share = shares[l]
        # final-month per-agent snapshot restricted to this location, best_idx run
        mask = loc_index[best_idx] == l
        n_in_loc = max(1, mask.sum())
        location_results.append({
            "name": loc.get("name", f"Location {l}"),
            "population": float(cfg["population"] * pop_share),
            "adoptionPct": float((enrolled[best_idx][mask].sum() / n_in_loc) * 100) if mask.any() else 0.0,
            "litterRatePct": float(litterRate_i[best_idx][mask].mean() * 100) if mask.any() else 0.0,
            "properRatePct": float((1 - litterRate_i[best_idx][mask]).mean() * 100) if mask.any() else 0.0,
            "cumulativeReports": float(loc_cum["reportsRaw"][l, best_idx]),
            "cumulativeVerified": float(loc_cum["verified"][l, best_idx]),
            "cumulativeRecovered": float(loc_cum["collectedThisMonth"][l, best_idx]),
        })

    # --- validation / sanity checks (section 17) ---
    warnings = []
    def check(cond, msg):
        if not cond:
            warnings.append(msg)
    import math
    flat_check_vals = [summary[k]["med"] for k in summary]
    check(all(math.isfinite(v) for v in flat_check_vals), "Non-finite (NaN/Infinity) value found in summary statistics.")
    check(summary["finalAdoptionPct"]["med"] <= 100.0001, "Adoption exceeded 100%.")
    check(summary["finalBiometricPct"]["med"] <= summary["finalAdoptionPct"]["med"] + 0.01, "Biometric adoption exceeded overall TTP adoption.")
    check(summary["totalRecycled"]["med"] <= summary["totalRecovered"]["med"] + 1e-6, "Recycled total exceeded recovered total.")
    check(summary["totalConfirmed"]["med"] <= summary["totalVerified"]["med"] + 1e-6, "Confirmed violations exceeded verified violations.")
    check(all(v >= -1e-6 for v in flat_check_vals), "A negative value was found in summary statistics.")

    return {
        "perMonth": per_month, "summary": summary, "medianRun": median_run, "numRuns": R,
        "locationResults": location_results, "warnings": warnings,
        "seedUsed": cfg["seed"], "resolvedCfg": cfg,
    }


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    payload = request.get_json(force=True)
    cfg = payload.get("cfg", {})
    base_params = payload.get("baseParams") or DEFAULT_PARAMS
    try:
        result = run_simulation(cfg, base_params)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Sensitivity analysis (section 16)
# ---------------------------------------------------------------------------

SENSITIVITY_PARAMS = [
    ("adoptionCeiling", "cfg", 0.7, 1.3, "Adoption ceiling"),
    ("deterrenceEffect", "base", 0.7, 1.3, "Deterrence effect"),
    ("reportDetectionRate", "base", 0.7, 1.3, "Reporting probability"),
    ("fineAccuracy", "base", 0.85, 1.1, "Verification accuracy"),
    ("qrFailureRate", "cfg", 0.5, 1.8, "QR failure rate (inverse of reliability)"),
    ("litterRecoveryRate", "cfg", 0.6, 1.3, "Litter recovery rate"),
    ("recyclingRate", "cfg", 0.7, 1.2, "Recycling rate"),
    ("heterogeneity", "cfg", 0.5, 1.5, "Behavioural heterogeneity"),
    ("transferRate", "cfg", 0.3, 2.0, "Ownership transfer rate"),
    ("biometricCeiling", "cfg", 0.5, 1.4, "Biometric adoption ceiling"),
]


@app.route("/api/sensitivity", methods=["POST"])
def api_sensitivity():
    payload = request.get_json(force=True)
    cfg = dict(payload.get("cfg", {}))
    base_params = dict(payload.get("baseParams") or DEFAULT_PARAMS)
    # Use a reduced, fixed-seed run for speed & comparability across sweeps
    cfg["agentSample"] = min(cfg.get("agentSample", 3000), 600)
    cfg["numRuns"] = min(cfg.get("numRuns", 30), 12)
    cfg["seed"] = cfg.get("seed") or 12345

    try:
        baseline = run_simulation(cfg, base_params)
        baseline_drop = baseline["summary"]["litterDropPct"]["med"]

        rows = []
        for key, target, lo_mult, hi_mult, label in SENSITIVITY_PARAMS:
            def run_with(mult):
                c2, b2 = dict(cfg), dict(base_params)
                if target == "cfg":
                    c2[key] = max(0.0, c2.get(key, 0) * mult) if key != "adoptionCeiling" else min(1.0, max(0.0, c2.get(key, 0.5) * mult))
                else:
                    b2[key] = max(0.0, min(0.99, b2.get(key, DEFAULT_PARAMS.get(key, 0.5)) * mult))
                return run_simulation(c2, b2)["summary"]["litterDropPct"]["med"]

            low_val = run_with(lo_mult)
            high_val = run_with(hi_mult)
            rows.append({
                "parameter": label, "key": key,
                "low": low_val, "baseline": baseline_drop, "high": high_val,
                "impact": abs(high_val - low_val),
            })
        rows.sort(key=lambda r: r["impact"], reverse=True)
        return jsonify({"baselineLitterDropPct": baseline_drop, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Scenario comparison (section 14)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {"key": "none", "label": "No TTP", "adoptionCeiling": 0.0},
    {"key": "low", "label": "Low adoption", "adoptionCeiling": 0.25},
    {"key": "moderate", "label": "Moderate adoption", "adoptionCeiling": 0.50},
    {"key": "high", "label": "High adoption", "adoptionCeiling": 0.75},
    {"key": "statewide", "label": "Statewide", "adoptionCeiling": 0.60, "population": 3000000},
    {"key": "campus", "label": "Campus pilot", "adoptionCeiling": 0.80, "population": 5000, "perCapita": 3},
]


@app.route("/api/compare", methods=["POST"])
def api_compare():
    payload = request.get_json(force=True)
    base_cfg = dict(payload.get("cfg", {}))
    base_params = dict(payload.get("baseParams") or DEFAULT_PARAMS)
    base_cfg["agentSample"] = min(base_cfg.get("agentSample", 3000), 1000)
    base_cfg["numRuns"] = min(base_cfg.get("numRuns", 30), 15)
    seed = base_cfg.get("seed") or 42

    rows = []
    try:
        for s in SCENARIOS:
            c2 = dict(base_cfg)
            c2["adoptionCeiling"] = s["adoptionCeiling"]
            if "population" in s:
                c2["population"] = s["population"]
            if "perCapita" in s:
                c2["perCapita"] = s["perCapita"]
            c2["seed"] = seed  # same seed across scenarios so differences are attributable to adoption, not RNG
            res = run_simulation(c2, base_params)
            s_sum = res["summary"]
            rows.append({
                "scenario": s["label"],
                "finalAdoptionPct": s_sum["finalAdoptionPct"]["med"],
                "litterDropPct": s_sum["litterDropPct"]["med"],
                "totalReportsRaw": s_sum["totalReportsRaw"]["med"],
                "totalConfirmed": s_sum["totalConfirmed"]["med"],
                "totalRecovered": s_sum["totalRecovered"]["med"],
                "totalRecycled": s_sum["totalRecycled"]["med"],
                "avgQrReliabilityPct": s_sum["avgQrReliabilityPct"]["med"],
                "finalBiometricPct": s_sum["finalBiometricPct"]["med"],
            })
        return jsonify({"rows": rows, "seedUsed": seed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug")
def debug():
    import os
    return jsonify({
        "root_path": app.root_path,
        "static_folder": app.static_folder,
        "static_folder_exists": os.path.isdir(app.static_folder),
        "static_contents": os.listdir(app.static_folder) if os.path.isdir(app.static_folder) else "MISSING",
        "root_contents": os.listdir(app.root_path),
    })


def find_index_file():
    """Case-insensitive lookup so a file uploaded as index.HTML (or any
    other casing) via a mobile GitHub upload still gets served correctly."""
    import os
    try:
        for fname in os.listdir(app.static_folder):
            if fname.lower() == "index.html":
                return fname
    except FileNotFoundError:
        pass
    return "index.html"


@app.route("/")
def index():
    return send_from_directory(app.static_folder, find_index_file())


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"TTP Impact Simulator running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
