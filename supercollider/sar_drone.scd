// ─────────────────────────────────────────────────────────────
// S.A.R  —  Seismic Activity Drone  (SuperCollider)
// ─────────────────────────────────────────────────────────────
// A GEOLOGICAL drone.  Sensor data shapes the CHARACTER of what
// you hear, not just volume.  Changes happen on a timescale of
// minutes, not seconds.  The drone has distinct regions of
// behaviour depending on seismic activity — from dark, minimal
// root tones to a full, breathing harmonic field.
//
// Design philosophy:
//   • GLACIAL tempo — chord progression takes 15-40 minutes to
//     cycle.  Individual chords linger for 5+ minutes.  20s
//     portamento makes changes feel tectonic, not musical.
//   • Sensor data is COMPOSITIONAL — activity controls which
//     voices are present (thresholds, not linear ramps).
//     Kp shapes timbre.  Events drive harmonic rhythm.
//   • Volume is nearly constant — the interest comes from
//     timbral and harmonic evolution, not loudness changes.
//   • Breathing layers operate on 1-4 minute cycles, creating
//     swells so slow they're felt more than heard.
//
// Chord progression (circle of 4ths):
//   Am → Dm → Gm → Cm → Fm → Bbm → Ebm  (7 states, looping)
//
// OSC:  /sar/drone/activity, /kp, /dst, /events, /alert
//
// Start with:  sclang supercollider/sar_drone.scd
// ─────────────────────────────────────────────────────────────

(
s.waitForBoot({

    // ── SynthDef ────────────────────────────────────────────
    SynthDef(\sar_drone, {
        arg out = 0,
            activity = 0.0,   // 0–1  max tile score
            kp       = 1.0,   // 0–9  geomagnetic index
            dst      = 0.0,   // nT   (negative = storm)
            events   = 0.0,   // active event count
            alert    = 0.0;   // 0–1  fusion alert level

        // ── All variable declarations ──
        var actL, kpL, dstL, evL, alertL;
        var actW, kpW, dstW, evW, alertW;
        var breath1, breath2, breath3, breath4, breath5;
        var progressRate, autoLinear, autoWander, autoPhase;
        var sensorShift, chordFloat, chordIdx;
        var rootNote, thirdNote, fifthNote, seventhNote, alertNote;
        var rootFreq, thirdFreq, fifthFreq, seventhFreq;
        var detuneAmt;
        var drift1, drift2, drift3, drift4, drift5;
        var vBal1, vBal2, vBal3, vBalSub, vBal7;
        var v1a, v1b, v1, v2, v3, vSub, v7th, vAlert;
        var sig, grainSig, amp, ampBreath;
        var filtFreq, filtQ;
        var delSig, delTime1, delTime2, delFb, delMod;
        var revRoom, revDamp, revMix;
        var panPos;

        // ─────────────────────────────────────────────────────
        // ── SENSOR ENVELOPES — geological time ──
        // ─────────────────────────────────────────────────────
        // Very long lags (20-30s) so sensor changes are felt
        // as slow tidal shifts, never as ramps.  Subtle wandering
        // keeps the drone alive between sensor poll intervals.

        actL   = Lag.kr(activity, 25);
        kpL    = Lag.kr(kp, 30);
        dstL   = Lag.kr(dst, 20);
        evL    = Lag.kr(events, 20);
        alertL = Lag.kr(alert, 15);

        // Gentle stochastic wandering (very subtle orbiting)
        actW   = (actL   + LFNoise2.kr(0.015).range(-0.03, 0.03)).clip(0, 1);
        kpW    = (kpL    + LFNoise2.kr(0.012).range(-0.2, 0.2)).clip(0, 9);
        dstW   = dstL    + LFNoise2.kr(0.018).range(-2, 2);
        evW    = (evL    + LFNoise2.kr(0.01).range(-2, 2)).max(0);
        alertW = (alertL + LFNoise2.kr(0.02).range(-0.02, 0.02)).clip(0, 1);

        // ─────────────────────────────────────────────────────
        // ── BREATHING — minutes-long cycles ──
        // ─────────────────────────────────────────────────────
        // Extremely slow, barely perceptible individually.
        // Their PRODUCT creates long, unpredictable swells.
        breath1 = SinOsc.kr(1/73.0).range(0.90, 1.0);     // ~73s
        breath2 = SinOsc.kr(1/127.0).range(0.92, 1.0);    // ~2min
        breath3 = LFNoise2.kr(1/180.0).range(0.88, 1.0);  // ~3min random
        breath4 = SinOsc.kr(1/97.0).range(0.93, 1.0);     // ~97s
        breath5 = LFNoise2.kr(1/240.0).range(0.85, 1.0);  // ~4min random

        // ─────────────────────────────────────────────────────
        // ── CHORD PROGRESSION — tectonic drift ──
        // ─────────────────────────────────────────────────────
        // Full cycle: ~40 min calm, ~15 min active.
        // Each chord lasts 5-6 minutes on average.
        // The random meander means the drone sometimes lingers
        // on a chord, sometimes drifts back, sometimes leaps.

        progressRate = evW.linlin(0, 100, 0.00042, 0.0011);
        autoLinear = LFSaw.kr(progressRate, 0).range(0, 6.999);

        // Very slow random meander — ±1 chord, glacially
        autoWander = LFNoise2.kr(0.003).range(-1.0, 1.0);
        autoPhase = (autoLinear + autoWander).mod(7);

        // Activity gently leans the chord forward (not a jump)
        sensorShift = actW.linlin(0, 1, 0, 1.5);
        chordFloat = (autoPhase + sensorShift).mod(7);
        chordIdx = chordFloat.round(1).min(6);

        // 20s portamento — chord changes are tectonic
        rootNote    = Lag.kr(Select.kr(chordIdx, #[45, 38, 43, 48, 41, 46, 39]), 20);
        thirdNote   = Lag.kr(Select.kr(chordIdx, #[48, 41, 46, 51, 44, 49, 42]), 20);
        fifthNote   = Lag.kr(Select.kr(chordIdx, #[52, 45, 50, 55, 48, 53, 46]), 20);
        seventhNote = Lag.kr(Select.kr(chordIdx, #[55, 48, 53, 58, 51, 56, 49]), 20);

        rootFreq    = rootNote.midicps;
        thirdFreq   = thirdNote.midicps;
        fifthFreq   = fifthNote.midicps;
        seventhFreq = seventhNote.midicps;

        // Dst detuning (storms add microtonal dissonance)
        detuneAmt = dstW.abs.linlin(0, 100, 0.0, 0.012).min(0.012);

        // Per-voice frequency drift (sub-0.1 Hz — heard as warmth)
        drift1 = LFNoise1.kr(0.04).range(-0.002, 0.002);
        drift2 = LFNoise1.kr(0.03).range(-0.003, 0.003);
        drift3 = LFNoise1.kr(0.05).range(-0.002, 0.002);
        drift4 = LFNoise1.kr(0.025).range(-0.0015, 0.0015);
        drift5 = LFNoise1.kr(0.035).range(-0.002, 0.002);

        // ─────────────────────────────────────────────────────
        // ── VOICE BALANCE — slow timbral shifts ──
        // ─────────────────────────────────────────────────────
        // Each voice wanders in level over 1-3 minutes.
        // The overall texture is always evolving.
        vBal1   = LFNoise2.kr(0.008).range(0.75, 1.0);
        vBal2   = LFNoise2.kr(0.006).range(0.55, 1.0);
        vBal3   = LFNoise2.kr(0.009).range(0.45, 1.0);
        vBalSub = LFNoise2.kr(0.005).range(0.70, 1.0);
        vBal7   = LFNoise2.kr(0.007).range(0.60, 1.0);

        // ─────────────────────────────────────────────────────
        // ── VOICES — scored by sensor character ──
        // ─────────────────────────────────────────────────────
        // Sensor data shapes WHICH voices are present and their
        // character, not just how loud they are.  The drone has
        // distinct regions of behaviour:
        //   quiet  (act < 0.15): root + sub, dark and spacious
        //   awake  (act 0.15-0.4): third emerges, warmth builds
        //   active (act 0.4-0.7): fifth + grains, open filter
        //   tense  (act > 0.7): full voicing, 7th, alert tones

        // Root — always present, the foundation
        v1a = LFSaw.ar(rootFreq * (1 + detuneAmt + drift1), 0, 0.10);
        v1b = LFSaw.ar(rootFreq * (1 - detuneAmt - drift1), 0, 0.10);
        v1 = (v1a + v1b) * vBal1 * breath1;

        // Minor third — EMERGES with activity (not linearly,
        // but through a soft threshold at ~0.15)
        v2 = LFTri.ar(
            thirdFreq * (1 + (detuneAmt * 0.5) + drift2 + (v1 * 0.0006)),
            0,
            0.07 * actW.linlin(0.1, 0.35, 0.0, 1.0).clip(0, 1)
        ) * vBal2 * breath2;

        // Fifth — arrives later, opens the harmonic space
        v3 = SinOsc.ar(fifthFreq * (1 + drift3), 0,
            0.05 * actW.linlin(0.3, 0.6, 0.0, 0.85).clip(0, 0.85)
        ) * vBal3 * breath3;

        // Sub bass — always present but swell with activity
        vSub = SinOsc.ar(rootFreq * 0.5 * (1 + drift4), 0,
            actW.linlin(0, 0.5, 0.10, 0.18).clip(0.10, 0.18)
        ) * vBalSub * breath5;

        // Minor 7th — only with Kp > 4 AND activity > 0.3
        v7th = SinOsc.ar(seventhFreq * (1 + drift5), 0,
            (kpW.linlin(4, 7, 0.0, 0.04).max(0)
             * actW.linlin(0.25, 0.5, 0.0, 1.0).clip(0, 1))
        ) * vBal7 * breath4;

        // Alert b9 — only during real alerts (> 0.3)
        alertNote = rootNote + 13;
        vAlert = SinOsc.ar(alertNote.midicps * (1 + drift2.neg), 0,
            alertW.linlin(0.3, 0.8, 0.0, 0.03).max(0));

        // ─────────────────────────────────────────────────────
        // ── MIX & SHAPE ──
        // ─────────────────────────────────────────────────────

        sig = v1 + v2 + v3 + vSub + v7th + vAlert;

        // Filter: Kp opens the timbre, but through wandering
        // cutoff.  Low Kp = muffled darkness.  High Kp = the
        // drone "breathes in" — brighter, more open.
        filtFreq = kpW.linexp(0, 9, 200, 2200)
            * LFNoise2.kr(0.012).range(0.82, 1.18);
        filtQ = alertW.linlin(0, 1, 0.45, 0.20);
        sig = RLPF.ar(sig, filtFreq.clip(100, 5000), filtQ);

        // Grain texture — only when events > ~10, sparse
        grainSig = Dust.ar(evW.linlin(8, 80, 0.2, 5).max(0.1)) * 0.025
            * BPF.ar(WhiteNoise.ar, Lag.kr(thirdFreq * 2, 8), 0.15)
            * actW.linlin(0.2, 0.5, 0.0, 1.0).clip(0, 1);
        sig = sig + grainSig;

        // Master amplitude: VERY tight range (0.12 - 0.24).
        // The volume barely changes — the interest comes from
        // timbral evolution, not loudness.  The breathing layers
        // create subtle, slow swells.
        ampBreath = (breath1 * breath2);
        amp = actW.linexp(0, 1, 0.12, 0.24) * ampBreath;
        sig = sig * amp;

        // ─────────────────────────────────────────────────────
        // ── DELAY — very slow spatial evolution ──
        // ─────────────────────────────────────────────────────
        delTime1 = kpW.linlin(0, 9, 0.38, 0.18);
        delTime2 = delTime1 * 0.618;

        delMod = (SinOsc.kr(0.04) * dstW.abs.linlin(0, 100, 0.0, 0.015))
            + LFNoise2.kr(0.015).range(-0.005, 0.005);
        delTime1 = (delTime1 + delMod).clip(0.05, 0.48);

        delFb = actW.linlin(0, 0.6, 0.20, 0.45).clip(0.20, 0.45);
        delSig = AllpassC.ar(sig, 0.5, delTime1, delFb * 3.0);
        delSig = AllpassC.ar(delSig, 0.5, delTime2, delFb * 2.0);
        sig = sig + (delSig * 0.25);

        // Very slow stereo drift
        panPos = LFNoise2.kr(0.008).range(-0.5, 0.5);
        sig = Pan2.ar(sig, panPos);

        // ─────────────────────────────────────────────────────
        // ── REVERB — slow-breathing space ──
        // ─────────────────────────────────────────────────────
        revRoom = alertW.linlin(0, 1, 0.50, 0.88)
            * LFNoise2.kr(0.006).range(0.90, 1.0);
        revDamp = kpW.linlin(0, 9, 0.70, 0.30)
            * LFNoise2.kr(0.005).range(0.92, 1.0);
        revMix = actW.linlin(0, 0.5, 0.20, 0.35).clip(0.20, 0.35)
            * breath3;

        sig = FreeVerb2.ar(sig[0], sig[1], revMix, revRoom.clip(0,1), revDamp.clip(0,1));

        sig = Limiter.ar(sig, 0.7);
        Out.ar(out, sig);
    }).add;

    s.sync;

    ">> S.A.R drone SynthDef loaded — minor chord evolution.".postln;

    // ── Launch the drone ────────────────────────────────────
    ~drone = Synth(\sar_drone, [
        \activity, 0.0,
        \kp,       1.0,
        \dst,      0.0,
        \events,   0.0,
        \alert,    0.0,
    ]);

    ">> S.A.R drone running — Am → Dm → Gm → Cm → Fm → Bbm → Ebm".postln;

    // ── OSC responders ──────────────────────────────────────
    // Cache last-received values for logging (Synth.get is async)
    ~lastPrint = 0;
    ~cachedKp = 1.0;
    ~cachedDst = 0.0;
    ~cachedEvents = 0;
    ~cachedActivity = 0.0;
    ~cachedAlert = 0.0;

    ~chordNames = ["Am", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"];

    OSCdef(\sar_activity, { |msg|
        var val = msg[1].asFloat;
        ~cachedActivity = val;
        ~drone.set(\activity, val);
        if (Main.elapsedTime - ~lastPrint > 15, {
            (">> OSC: activity=" ++ val.round(0.001)
             ++ "  kp=" ++ ~cachedKp.round(0.1)
             ++ "  events=" ++ ~cachedEvents
            ).postln;
            ~lastPrint = Main.elapsedTime;
        });
    }, '/sar/drone/activity');

    OSCdef(\sar_kp, { |msg|
        var val = msg[1].asFloat;
        ~cachedKp = val;
        ~drone.set(\kp, val);
        (">> OSC: kp=" ++ val.round(0.1)
         ++ (if (val > 3, {" — minor 7th active"}, {""}))
        ).postln;
    }, '/sar/drone/kp');

    OSCdef(\sar_dst, { |msg|
        var val = msg[1].asFloat;
        ~cachedDst = val;
        ~drone.set(\dst, val);
        (">> OSC: dst=" ++ val.round(1) ++ " nT"
         ++ (if (val.abs > 30, {" — detuning active"}, {""}))
        ).postln;
    }, '/sar/drone/dst');

    OSCdef(\sar_events, { |msg|
        var val = msg[1].asFloat;
        ~cachedEvents = val;
        ~drone.set(\events, val);
    }, '/sar/drone/events');

    OSCdef(\sar_alert, { |msg|
        var val = msg[1].asFloat;
        ~cachedAlert = val;
        ~drone.set(\alert, val);
        if (val > 0.1, {
            (">> OSC: ALERT level=" ++ val.round(0.01)
             ++ " — tension tone active").postln;
        });
    }, '/sar/drone/alert');

    ">> S.A.R OSC responders active on port 57120.".postln;
    ">> Drone evolves autonomously through minor keys.".postln;
    ">> Sensor data accelerates and colours the progression.".postln;
});
)
