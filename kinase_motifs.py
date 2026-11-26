"""
Locate the ATP pocket of a protein kinase from sequence alone.

Both benchmarks are kinase panels: Davis is 442 kinase assays and KIBA is
the Kinase Inhibitor BioActivity set. The ATP-binding site of the
eukaryotic protein kinase fold is the most conserved pocket in the
proteome and is identifiable from primary sequence by five motifs that
appear in a fixed order. That makes a biological reference available
without downloading a structure database, which matters here for two
reasons: the result is reproducible from the repository alone, and there
is no database release to pin a number to.

The motifs, in order along the catalytic domain:

  G-loop    GxGxxG     beta1-beta2, wraps the ATP phosphates
  beta3 Lys VAIK       invariant lysine, contacts the alpha/beta phosphates
  hinge                between beta5 and alphaD, where type-I inhibitors sit
  HRD                  catalytic loop aspartate
  DFG                  start of the activation loop, the magnesium chelator

An ATP-competitive inhibitor contacts the adenine pocket bounded by the
G-loop, the hinge and the DFG motif, plus the gatekeeper. We therefore
define the pocket as windows around the detected motifs rather than as
the motif residues themselves, since a contact two residues from the DFG
aspartate is in the pocket and a motif-exact definition would score it as
a miss.

What this is not: a substitute for co-crystal contacts. A motif-derived
pocket is a prior, not a measurement, and the enrichment it supports is
"the selected contacts concentrate on the catalytic machinery" rather
than "the selected contacts are the crystallographic contacts". Any paper
using this must say so.

Detection is deliberately conservative. A sequence whose motifs are not
found in the right order is reported as undetected rather than assigned a
guessed pocket, because a wrong pocket inflates or deflates enrichment
with no way to tell which. Call audit() on a dataset before trusting any
enrichment computed from it.
"""
import re
from collections import Counter

# Pocket half-width around each motif, in residues. The adenine pocket of
# the ePK fold spans roughly 10-15 residues either side of the hinge and
# the DFG motif; the G-loop contributes its own length plus a few.
WINDOW = {"gloop": 6, "vaik": 5, "hinge": 8, "hrd": 6, "dfg": 8}

# Ordering constraints, in residues, from the ePK catalytic domain. These
# are loose bounds meant to reject false positives, not to model geometry.
SPACING = {("gloop", "vaik"): (8, 60),
           ("vaik", "hrd"): (60, 220),
           ("hrd", "dfg"): (10, 45)}

# Requiring all five motifs rejects real kinases. Measured on the KIBA
# panel, 229 sequences: all five resolves 60.3%, with the beta3 lysine
# absent in 48 and HRD in 38. The lysine is named after PKA's VAIK but is
# poorly conserved at the flanking positions, so any tight pattern there
# throws away genuine catalytic domains.
#
# HRD and DFG are the two conserved elements, and their spacing in the
# ePK fold is tight (10-45 residues), which does the filtering that the
# other motifs were being asked to do. ANCHORS must be present; the rest
# extend the pocket when found and are skipped when not.
ANCHORS = ("hrd", "dfg")

PATTERNS = {
    # GxGxxG, the classic phosphate-binding loop. GxGxxA and GxGxxS occur
    # in a minority of kinases and are included.
    "gloop": re.compile(r"G.G..[GAS]"),
    # The beta3 lysine. VAIK is the textbook spelling; the first and third
    # positions tolerate the other small hydrophobics.
    "vaik": re.compile(r"[VILMA][AGSC][VILMA]K"),
    # Catalytic loop. HRD dominates; HCD, HGD and HKD occur in tyrosine
    # kinases and in a few atypical families.
    "hrd": re.compile(r"H[RCGKAS]D"),
    # Activation loop start. DFG dominates; DLG, DWG and DFS are the known
    # variants in human kinases.
    "dfg": re.compile(r"D[FLWY][GS]"),
}


def _candidates(seq, name):
    return [m.start() for m in PATTERNS[name].finditer(seq)]


def find_motifs(seq, strict=False):
    """Return {motif: index} for a kinase domain, or None if not resolved.

    Anchored on HRD and DFG: the least ambiguous motifs, the closest
    together, and the only two required. The beta3 lysine and the G-loop
    extend the pocket when they are found at a plausible spacing and are
    omitted otherwise, so a kinase with an unusual beta3 strand is scored
    on the catalytic and activation loops rather than discarded.

    Choosing the globally best-spaced combination rather than the first
    match matters: GxGxxG occurs by chance roughly once per 400 residues,
    so a first-match rule picks a spurious G-loop on most sequences of
    this length.

    strict=True restores the all-five requirement, for reporting the
    detection rate both ways.
    """
    seq = seq.upper()
    hrds, dfgs = _candidates(seq, "hrd"), _candidates(seq, "dfg")
    gloops, vaiks = _candidates(seq, "gloop"), _candidates(seq, "vaik")
    if not (hrds and dfgs):
        return None
    if strict and not (gloops and vaiks):
        return None

    lo, hi = SPACING[("hrd", "dfg")]
    best = None
    for h in hrds:
        for d in dfgs:
            if not (lo <= d - h <= hi):
                continue
            m = {"hrd": h, "dfg": d}
            klo, khi = SPACING[("vaik", "hrd")]
            ks = [k for k in vaiks if klo <= h - k <= khi]
            if ks:
                k = max(ks)                  # the one closest to HRD
                m["vaik"] = k
                glo, ghi = SPACING[("gloop", "vaik")]
                gs = [g for g in gloops if glo <= k - g <= ghi]
                if gs:
                    m["gloop"] = max(gs)     # the one closest to beta3
                # The hinge is not a sequence motif. It lies between beta5
                # and alphaD, reliably about halfway from the beta3 lysine
                # to the catalytic loop, so it is only placed when the
                # lysine anchoring it was found.
                m["hinge"] = k + int(0.45 * (h - k))
            elif strict:
                continue
            # Prefer the tightest HRD-DFG pair, which in the ePK fold is
            # the catalytic loop and not a coincidental repeat.
            if best is None or (d - h) < best[0]:
                best = (d - h, m)
    if best is None:
        return None
    if strict and len(best[1]) < 5:
        return None
    return best[1]


def pocket_residues(seq, motifs=None, strict=False):
    """0-based residue indices forming the ATP pocket, or None.

    Built only from the motifs actually located, so a sequence resolved on
    the anchors alone gets a smaller pocket rather than an invented one.
    Enrichment is computed against each protein's own pocket fraction, so
    a smaller pocket raises that protein's chance level instead of
    flattering it.
    """
    motifs = (motifs if motifs is not None
              else find_motifs(seq, strict=strict))
    if motifs is None:
        return None
    L = len(seq)
    res = set()
    for name, pos in motifs.items():
        w = WINDOW[name]
        res.update(range(max(0, pos - w), min(L, pos + w + 1)))
    return res


def audit(sequences, verbose=True, strict=False):
    """Detection rate and pocket sizes over a collection of sequences.

    Run this before computing any enrichment. A detection rate below
    roughly 0.7 means the motif model does not fit the panel and the
    enrichment would be computed on a biased subset of it.
    """
    n, found, sizes, fracs, miss = 0, 0, [], [], Counter()
    partial = Counter()
    for s in sequences:
        n += 1
        m = find_motifs(s, strict=strict)
        if m is None:
            for name in PATTERNS:
                if not _candidates(s.upper(), name):
                    miss[name] += 1
            continue
        found += 1
        partial[len(m)] += 1
        p = pocket_residues(s, m)
        sizes.append(len(p))
        fracs.append(len(p) / len(s))
    out = {"n": n, "detected": found, "rate": found / n if n else 0.0,
           "mean_pocket": sum(sizes) / len(sizes) if sizes else 0.0,
           "mean_pocket_frac": sum(fracs) / len(fracs) if fracs else 0.0,
           "absent_motif": dict(miss), "motifs_found": dict(partial)}
    if verbose:
        print(f"  sequences                {out['n']}")
        print(f"  kinase domain resolved   {out['detected']} "
              f"({out['rate']:.1%})")
        print(f"  mean pocket size         {out['mean_pocket']:.1f} residues")
        print(f"  mean pocket fraction     {out['mean_pocket_frac']:.3f} "
              f"of the sequence  <- chance level for enrichment")
        if partial:
            print(f"  motifs located per sequence  "
                  f"{dict(sorted(partial.items()))}  "
                  f"(5 = all; 2 = anchors only)")
        if miss:
            print(f"  motif never found in the undetected ones: "
                  f"{out['absent_motif']}")
        if out["rate"] < 0.7:
            print(f"\n  NOTE: {out['rate']:.1%} of this panel resolves, so "
                  f"the detected subset is conditioned on detectability.\n"
                  f"  The ABSOLUTE enrichment is therefore a statement "
                  f"about that subset, not the panel, and should be\n"
                  f"  reported with the rate beside it. The PAIRED "
                  f"comparison between two decoders stays valid, since\n"
                  f"  both are scored on the same sequences; that is the "
                  f"comparison to lead with.")
    return out
