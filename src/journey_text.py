"""
journey_text.py — YOUR words for the journey.

Fill in the strings below (plain text or markdown). Whatever is left empty is simply
not shown in the app.
"""

TEXT = {
    "intro": (
        "GENOME turns news articles into events - who did what to whom, and when - and tags every event on two axes: "
        "verbal vs. material, and cooperation vs. conflict. That split is the whole premise of this app: once words "
        "and deeds are separated, everything below is really one question asked at a different scale each time - "
        "does a country's rhetoric point the same way as its behaviour?\n\n"
        "Section A starts at the widest possible lens: which country pairs have the most conflict between them, full "
        "stop (US-Yemen at a 93.6% hostile share, 339 events; Russia-Ukraine at 89.9%, 3,162 events). That ranking "
        "is not decorative - it is what picks the conflicts worth following through every section after this one. A "
        "conflict with too little data, or too little actual conflict in it, would not carry a reliable signal "
        "through the rest of the app.\n\n"
        "The app as a whole moves from **describing** (here, and section 3) through **structure** (section 2) to "
        "**testing** (section 4) and **applying** (section 5). Every chart carries a 'how to read this' note "
        "repeating the same two things: this is reported events, not proven flows, and a null or weak result is "
        "still a result, not a failure of the method."
    ),
    "movement": (
        "A single period-wide side score hides a lot: a country can start firmly on one side and drift, or swing "
        "back and forth, and a period average would show neither. The 12-week rolling window exists specifically "
        "to catch that movement - each line is the same side score used elsewhere in the app, just recomputed every "
        "week instead of once. The axis runs from Russia (negative) to Ukraine (positive) for this conflict; the "
        "grey band is 'no clear side', and the gaps are not neutrality - they are too few events in that window to "
        "say anything. Only the biggest players (the US, China, India and a handful of others) have enough weekly "
        "volume for an unbroken line, which is itself a finding about who GENOME has enough signal on."
    ),
    "side_all": (
        "Section C takes the period-wide side score and turns it into one number per country, so 45 countries "
        "(everyone with at least 3 events toward both parties) can be compared at a glance instead of one at a "
        "time. The toggle that adds words-only and actions-only positions on top of each bar exists because the "
        "combined score can itself mislead: a country with hostile words and friendly actions can land on a bar "
        "that just looks 'moderate', when its words and deeds are actually pulling in opposite directions and "
        "cancelling out.\n\n"
        "That is exactly what section D checks directly, by plotting verbal stance against material stance for "
        "every country at once instead of folding them into one score. Of 57 countries plotted, 45 sit close to the "
        "diagonal - words and actions agree - and only one is a real mismatch: South Korea toward Ukraine, hostile "
        "in words but friendly in actions. One mismatch out of 57 is informative on its own: for this conflict, the "
        "'say one thing, do another' pattern this app is built to catch turns out to be the exception, not the "
        "rule - which only sections C and D together could show."
    ),
    "clusters_intro": (
        "Sections A through D look at countries one at a time, toward one conflict. But countries also cooperate "
        "with each other more broadly, and the shape of who-cooperates-with-whom is a signal independent of any "
        "single conflict's side-taking. This section asks whether countries sort into blocks, and if a block exists "
        "in what countries say, whether the same block exists in what they do.\n\n"
        "The one result worth remembering: the two groupings disagree. Actions sort the world into 4 blocks (33 of "
        "60 countries placed with enough certainty); words sort it into 3 blocks (39 certain), and the two line up "
        "only partly."
    ),
    "clusters_map": (
        "Side by side, the two maps show the same countries grouped by actions and by words. Colour is the block; "
        "a marked dot is a country whose word-block and action-block disagree - that detail is worth more than the "
        "blocks themselves. Filtering by period, resolution, how events are counted, and whether to exclude the "
        "largest countries exists so a block isn't accidentally just 'the countries closest to the biggest "
        "player' - that has to survive being checked, not be assumed."
    ),
    "clusters_brokers": (
        "Blocks describe where most of a country's cooperation goes; brokers are the exception that connect blocks "
        "to each other instead of sitting inside one. The bridge-score table (UAE, Turkey, Kuwait stand out here) is "
        "corrected against chance, because any country with many partners looks like a 'bridge' by volume alone - "
        "the correction is what separates an actual broker from a country that is simply active.\n\n"
        "The flows-between-blocks heatmap is the block-level version of the same question: not which countries "
        "bridge, but how much cooperation crosses block lines at all. And the adversaries check - 29% of conflict "
        "events fall inside the same cooperation block, against 26% by chance - answers something the maps alone "
        "can't: these blocks are not simply 'sides' of the world's conflicts. Neighbours cooperate and fight the "
        "same neighbours; cooperation blocks are regional, not alliance camps."
    ),
    "clusters_words": (
        "This is the group-level version of section D's words/actions question. Where words and actions assign a "
        "country to different blocks, that country's rhetoric is not just imprecise about one relationship - it "
        "points toward an entirely different set of partners than its behaviour does. Agreement between the two "
        "groupings is modest (adjusted Rand index 0.22, 95% interval 0.14-0.36 against a permutation null of 0.00), "
        "and whether earlier words-based blocks predict later material cooperation is tested directly rather than "
        "assumed - the honest answer is that they predict weakly. 'Candidate latent tie' is used instead of "
        "'hidden tie' for the same reason section D's one mismatch mattered more than the 56 agreements: a weak or "
        "absent prediction is the result, not a failure of the method."
    ),
    "clusters_sure": (
        "Every block above comes from one clustering run, and clustering is sensitive to exactly how the "
        "underlying graph is built. Before trusting any block, the pipeline is rerun 100 times on resampled data; a "
        "country only counts as 'placed' if it lands in the same block at least 70% of the time. Comparing against "
        "a null model (a random graph with the same in/out degree per country) is what turns 'there is a block "
        "here' into 'there is a block here that is not just what volume alone would produce'. Removing the largest "
        "country and re-clustering checks the last obvious failure mode: that the whole pattern is really just "
        "'everyone's relationship to one dominant country', dressed up as blocks."
    ),
    "clusters_time": (
        "Blocks are not static, and a country changing block over time is a different kind of signal than a "
        "country disagreeing between words and actions at one point - it is about alignment moving, not about "
        "being inconsistent. This view exists so a reader can see realignment happening, instead of only seeing the "
        "end state the other tabs show."
    ),
    "clusters_chains": (
        "Blocks say who cooperates with whom in aggregate; they don't say whether that cooperation is direct or "
        "passed along through others. This tab connects back to section 4's chain analysis - placing the chains "
        "inside the block structure shows whether support mostly moves within a block (reinforcing it) or crosses "
        "between blocks, which would make a block boundary look firmer on the map than it actually is in practice."
    ),
    "chains_intro": (
        "'Which side does a country take' and 'who does a country actually pass support to' are different "
        "questions - a country can be firmly on one side without ever appearing as a direct link in a support "
        "chain, if its help is reported as going through an intermediary instead. This section moves from "
        "side-taking to the chains themselves: A helps B, B helps C, sometimes ending in B attacking D - reported "
        "paths, not proven transfers, as the 'how to read this' note keeps repeating."
    ),
    "chains_map": (
        "The arc map shows the heaviest reported aid relationships, with the option to follow one country two "
        "steps back and two steps forward - the more useful view in practice, since the full map is too dense to "
        "read as a whole. The biggest gap in this map, worth stating plainly: organisations (EU, NATO, UN) have no "
        "country id in GENOME and cannot appear here at all, even though a large share of real aid is organised "
        "through them. What's shown is a map of countries, not of how aid actually moves."
    ),
    "chains_stats": (
        "A chain that looks 'coordinated' can simply be two busy countries doing a lot of things in the same "
        "period - a war week makes everyone look coordinated at once. That is why 1,168 tested chains are compared "
        "against a shuffled version of the same data that preserves each country's total activity, and in the "
        "stricter version only shuffles events into equally busy windows. 30 chains pass an uncorrected 1% "
        "threshold where chance alone would produce about 12 - but after correcting for testing over a thousand "
        "chains at once, 0 remain. That null result is reported here rather than hidden, because 'most apparent "
        "support chains are just volume' is the honest finding of this section, not a dead end."
    ),
    "chains_words": (
        "If support often arrives through a chain rather than directly, do words at least predict where new "
        "support will go? Country pairs with cooperative words early in the period get a new aid link later 31% of "
        "the time, against 17% for comparable pairs without those words - a real but modest effect (+14.5 points, "
        "95% interval +5.3 to +23.1), checked against an activity-balance test so it isn't just 'pairs with words "
        "talk to each other more in general'. Below that, pairs with words but no material link at all (France to "
        "the US: 49 word events, 0 material) are shown as candidate latent ties, not findings - a lead for the "
        "reader to judge, not a claim this app makes for them."
    ),
    "case_intro": (
        "The four cases are not an arbitrary top-4: three are chosen because they are the conflicts with the most "
        "reported aid in the data (Ukraine first, Palestine second, Iran among the top five), and the fourth is "
        "chosen by the data itself, as the pair whose words and actions disagree the most anywhere in the dataset. "
        "The point of pulling everything into one page per case is to see whether the separate angles - blocks, "
        "chains, side-taking - actually agree with each other once you stop looking at them one at a time."
    ),
    "case_ukraine": (
        "Here the blocks (section 2), the bridges (UK, Germany, Canada, France) and the chains (section 4) all "
        "describe the same relationship from different angles, and the quarterly tables show the block assignment "
        "is not fixed - it moves. The supplier Sankey answers the question a policymaker would actually ask - who "
        "supplies Ukraine, and through whom - and the events behind any one path are one click away, specifically "
        "so a surprising-looking path can be checked against what was actually reported, not just trusted because "
        "the diagram shows it."
    ),
    "case_iran": (
        "The Iran-Israel case is where the aid-then-attack pattern from section 4 is most visible: the chains that "
        "survive at 2-week windows cluster around Iran as a hub. That they do not hold at 4- or 8-week windows is "
        "exactly why this page treats them as a lead to look into, not a finding - a pattern that disappears when "
        "you change the window length is usually measuring the window, not the world."
    ),
    "case_palestine": (
        "Palestine receives the second-most reported aid of any country in this dataset (373 events, after "
        "Ukraine's 620), and the heaviest single aid chain found anywhere in this analysis runs USA to Israel to "
        "Palestine. That one path is worth sitting with specifically because it is the clearest example of why "
        "this app insists on 'reported path', not 'proven flow': GENOME's AID category is wide enough to include "
        "humanitarian aid, military aid and exchanges of bodies or prisoners in the same bucket, so the path itself "
        "doesn't say which of those it is without reading the events behind it."
    ),
}

HINTS = {
    "intro": "Where you started: first the data without credits, then a first analysis to see whether a reliability factor is feasible.",
    "movement": "How you went from static scores to movement: why a rolling window, what the lines and the animation show about the US, China, India and others, and why short windows only work for big players.",
    "side_all": "How to read the chart: who leans where, and who sits in the middle (the 'strange ducks' that pick no side).",
    "clusters_intro": "Why you moved from pairs to blocks: what a block is (and is not), and the one result you most want a policymaker to remember.",
    "clusters_map": "What you see on the two maps: which blocks, where words and actions disagree, which countries are grey and why.",
    "clusters_brokers": "Who bridges the blocks and why that matters (and why a bridge score only counts above chance).",
    "clusters_words": "What it means that words and actions only partly agree; candidate latent ties versus empty words; why the prediction is weak.",
    "clusters_sure": "How much you trust the blocks: certainty, the comparison with chance, the effect of leaving out the biggest country.",
    "clusters_time": "Which countries change block, and when; what is behind it.",
    "clusters_chains": "How the chains fit inside the blocks; what you would do with them.",
    "chains_intro": "Why you moved from 'which side' to 'who passes support on to whom': what a chain is and what it is not.",
    "chains_map": "What you notice on the map: the biggest flows, the hubs, what is missing (organisations, meetings).",
    "chains_stats": "Why the comparison with chance matters, and what it means that most chains disappear after the correction.",
    "chains_words": "What the words-before-aid result does and does not tell you; why you call a words-only tie 'candidate', not 'hidden'.",
    "case_intro": "How you chose the cases and what you hope to see in them.",
    "case_ukraine": "What the supplier picture around Ukraine (or Russia) shows, and one path you looked at closely.",
    "case_iran": "What the networks around Iran and Israel show, and why the aid-then-attack chains need care.",
    "case_palestine": "What the picture around Palestine shows, and the path through Israel.",
}
