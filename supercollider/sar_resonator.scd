// ─────────────────────────────────────────────────────────────
// S.A.R  —  Sympathetic String Resonator  (SuperCollider)
// ─────────────────────────────────────────────────────────────
// Physical string excitation through speaker-to-string coupling.
// Drives harmonic banks at guitar-string fundamentals with
// micro-wobble and sparse impulses.  Designed to make real
// guitar strings vibrate sympathetically.
//
// Architecture:
//   6 strings (standard tuning): E2 A2 D3 G3 B3 E4
//   Each string → up to 8 partials (fundamental × [1..8])
//   Micro-wobble on every partial sweeps through the string's
//   resonance window.  Sparse impulses act as "plucks".
//
// Sensor mapping:
//   activity  → how many strings are active (E2+A2 always on)
//   kp        → harmonic richness (upper partials fade in)
//   dst       → inter-partial detuning (beating during storms)
//   events    → impulse density (more quakes = more plucks)
//   alert     → dissonant partials (7th, 11th harmonics)
//
// OSC:  /sar/resonator/activity, /kp, /dst, /events, /alert
//
// Standalone:  sclang supercollider/sar_resonator.scd
// ─────────────────────────────────────────────────────────────

(
s.waitForBoot({

    // ── SynthDef ────────────────────────────────────────────
    SynthDef(\sar_resonator, {
        arg out = 0,
            activity = 0.0,   // 0–1  max tile score
            kp       = 1.0,   // 0–9  geomagnetic index
            dst      = 0.0,   // nT   (negative = storm)
            events   = 0.0,   // active event count
            alert    = 0.0;   // 0–1  fusion alert level

        // ── Variable declarations ──
        var actL, kpL, dstL, evL, alertL;
        var actW, kpW, dstW, evW, alertW;
        var breath1, breath2;
        var stringFreqs, numStrings;
        var stringGates, partialDepth;
        var dstDetune;
        var sig, impulseSig, impulseRate;
        var nPartials, i, j;
        var strSig, partialSig, partFreq, wobbleRate, wobble, partAmp;
        var dissonantSig, disFreq1, disFreq2;
        var masterAmp;
        var panPos;

        // ─────────────────────────────────────────────────────
        // ── SENSOR ENVELOPES ──
        // ─────────────────────────────────────────────────────
        actL   = Lag.kr(activity, 20);
        kpL    = Lag.kr(kp, 25);
        dstL   = Lag.kr(dst, 15);
        evL    = Lag.kr(events, 15);
        alertL = Lag.kr(alert, 12);

        actW   = (actL   + LFNoise2.kr(0.018).range(-0.02, 0.02)).clip(0, 1);
        kpW    = (kpL    + LFNoise2.kr(0.014).range(-0.15, 0.15)).clip(0, 9);
        dstW   = dstL    + LFNoise2.kr(0.02).range(-1.5, 1.5);
        evW    = (evL    + LFNoise2.kr(0.012).range(-1.5, 1.5)).max(0);
        alertW = (alertL + LFNoise2.kr(0.025).range(-0.015, 0.015)).clip(0, 1);

        // ─────────────────────────────────────────────────────
        // ── BREATHING ──
        // ─────────────────────────────────────────────────────
        breath1 = SinOsc.kr(1/83.0).range(0.88, 1.0);
        breath2 = LFNoise2.kr(1/150.0).range(0.90, 1.0);

        // ─────────────────────────────────────────────────────
        // ── STRING FUNDAMENTALS (standard guitar tuning) ──
        // ─────────────────────────────────────────────────────
        // E2=82.41, A2=110, D3=146.83, G3=196, B3=246.94, E4=329.63
        stringFreqs = #[82.41, 110.0, 146.83, 196.0, 246.94, 329.63];
        numStrings = 6;

        // Activity controls which strings are active:
        //   0.0  → strings 0,1 (E2, A2)
        //   0.3  → + string 2 (D3)
        //   0.5  → + string 3 (G3)
        //   0.7  → + string 4 (B3)
        //   0.85 → + string 5 (E4)
        stringGates = [
            1.0,
            1.0,
            actW.linlin(0.15, 0.35, 0.0, 1.0).clip(0, 1),
            actW.linlin(0.35, 0.55, 0.0, 1.0).clip(0, 1),
            actW.linlin(0.55, 0.75, 0.0, 1.0).clip(0, 1),
            actW.linlin(0.75, 0.90, 0.0, 1.0).clip(0, 1),
        ];

        // Kp controls how many upper partials are audible (1–8)
        partialDepth = kpW.linlin(0, 9, 2, 8).round(1).clip(1, 8);

        // Dst controls inter-partial micro-detuning
        dstDetune = dstW.abs.linlin(0, 80, 0.0, 0.008).min(0.008);

        // Events control impulse density
        impulseRate = evW.linlin(0, 60, 0.08, 2.5).clip(0.05, 3.0);

        // ─────────────────────────────────────────────────────
        // ── HARMONIC BANK — 6 strings × up to 8 partials ──
        // ─────────────────────────────────────────────────────
        sig = DC.ar(0);

        // ── String 0 (E2 82.41 Hz) — always on ──
        6.do({ |j|
            var pNum = j + 1;
            var gate = (pNum <= partialDepth);
            var freq = stringFreqs[0] * pNum * (1 + (dstDetune * LFNoise1.kr(0.03 + (j * 0.007)).range(-1, 1)));
            var wRate = LFNoise2.kr(0.006 + (j * 0.002)).range(0.08, 0.4);
            var wob = SinOsc.kr(wRate).range(-1.2, 1.2);
            var amp = (0.035 / pNum.sqrt) * gate * stringGates[0]
                * LFNoise2.kr(0.004 + (j * 0.001)).range(0.7, 1.0);
            sig = sig + SinOsc.ar(freq + wob, 0, amp);
        });

        // ── String 1 (A2 110 Hz) — always on ──
        6.do({ |j|
            var pNum = j + 1;
            var gate = (pNum <= partialDepth);
            var freq = stringFreqs[1] * pNum * (1 + (dstDetune * LFNoise1.kr(0.035 + (j * 0.006)).range(-1, 1)));
            var wRate = LFNoise2.kr(0.007 + (j * 0.002)).range(0.1, 0.45);
            var wob = SinOsc.kr(wRate).range(-1.0, 1.0);
            var amp = (0.03 / pNum.sqrt) * gate * stringGates[1]
                * LFNoise2.kr(0.005 + (j * 0.001)).range(0.7, 1.0);
            sig = sig + SinOsc.ar(freq + wob, 0, amp);
        });

        // ── String 2 (D3 146.83 Hz) ──
        5.do({ |j|
            var pNum = j + 1;
            var gate = (pNum <= partialDepth);
            var freq = stringFreqs[2] * pNum * (1 + (dstDetune * LFNoise1.kr(0.032 + (j * 0.008)).range(-1, 1)));
            var wRate = LFNoise2.kr(0.008 + (j * 0.003)).range(0.12, 0.5);
            var wob = SinOsc.kr(wRate).range(-1.1, 1.1);
            var amp = (0.028 / pNum.sqrt) * gate * stringGates[2]
                * LFNoise2.kr(0.006 + (j * 0.001)).range(0.65, 1.0);
            sig = sig + SinOsc.ar(freq + wob, 0, amp);
        });

        // ── String 3 (G3 196 Hz) ──
        5.do({ |j|
            var pNum = j + 1;
            var gate = (pNum <= partialDepth);
            var freq = stringFreqs[3] * pNum * (1 + (dstDetune * LFNoise1.kr(0.028 + (j * 0.005)).range(-1, 1)));
            var wRate = LFNoise2.kr(0.009 + (j * 0.002)).range(0.1, 0.45);
            var wob = SinOsc.kr(wRate).range(-0.9, 0.9);
            var amp = (0.025 / pNum.sqrt) * gate * stringGates[3]
                * LFNoise2.kr(0.005 + (j * 0.002)).range(0.65, 1.0);
            sig = sig + SinOsc.ar(freq + wob, 0, amp);
        });

        // ── String 4 (B3 246.94 Hz) ──
        4.do({ |j|
            var pNum = j + 1;
            var gate = (pNum <= partialDepth);
            var freq = stringFreqs[4] * pNum * (1 + (dstDetune * LFNoise1.kr(0.025 + (j * 0.009)).range(-1, 1)));
            var wRate = LFNoise2.kr(0.007 + (j * 0.003)).range(0.15, 0.5);
            var wob = SinOsc.kr(wRate).range(-0.8, 0.8);
            var amp = (0.022 / pNum.sqrt) * gate * stringGates[4]
                * LFNoise2.kr(0.004 + (j * 0.002)).range(0.6, 1.0);
            sig = sig + SinOsc.ar(freq + wob, 0, amp);
        });

        // ── String 5 (E4 329.63 Hz) ──
        4.do({ |j|
            var pNum = j + 1;
            var gate = (pNum <= partialDepth);
            var freq = stringFreqs[5] * pNum * (1 + (dstDetune * LFNoise1.kr(0.022 + (j * 0.007)).range(-1, 1)));
            var wRate = LFNoise2.kr(0.008 + (j * 0.002)).range(0.12, 0.48);
            var wob = SinOsc.kr(wRate).range(-0.7, 0.7);
            var amp = (0.020 / pNum.sqrt) * gate * stringGates[5]
                * LFNoise2.kr(0.005 + (j * 0.001)).range(0.6, 1.0);
            sig = sig + SinOsc.ar(freq + wob, 0, amp);
        });

        // ─────────────────────────────────────────────────────
        // ── IMPULSE EXCITATION — sparse resonant plucks ──
        // ─────────────────────────────────────────────────────
        // Filtered noise bursts through resonant BPFs at string
        // fundamentals.  These "kick-start" string vibration.
        impulseSig = DC.ar(0);
        6.do({ |i|
            var trig = Dust.ar(impulseRate * stringGates[i] * LFNoise2.kr(0.01).range(0.3, 1.0));
            var burst = trig * WhiteNoise.ar(0.15);
            var reson = BPF.ar(burst, stringFreqs[i], 0.005);
            impulseSig = impulseSig + reson;
        });

        sig = sig + (impulseSig * evW.linlin(0, 30, 0.3, 1.0).clip(0.2, 1.0));

        // ─────────────────────────────────────────────────────
        // ── DISSONANT PARTIALS — alert tension ──
        // ─────────────────────────────────────────────────────
        // 7th and 11th harmonics of the lowest string, only
        // during elevated alert levels.
        disFreq1 = stringFreqs[0] * 7 * (1 + LFNoise1.kr(0.02).range(-0.003, 0.003));
        disFreq2 = stringFreqs[0] * 11 * (1 + LFNoise1.kr(0.015).range(-0.002, 0.002));
        dissonantSig = SinOsc.ar(disFreq1, 0, 0.012) + SinOsc.ar(disFreq2, 0, 0.008);
        dissonantSig = dissonantSig * alertW.linlin(0.25, 0.7, 0.0, 1.0).clip(0, 1);
        sig = sig + dissonantSig;

        // ─────────────────────────────────────────────────────
        // ── OUTPUT SHAPING ──
        // ─────────────────────────────────────────────────────
        masterAmp = actW.linexp(0, 1, 0.15, 0.35) * breath1 * breath2;
        sig = sig * masterAmp;

        // Gentle highpass to remove DC and sub-bass rumble
        sig = HPF.ar(sig, 60);

        // Very slow stereo drift
        panPos = LFNoise2.kr(0.005).range(-0.3, 0.3);
        sig = Pan2.ar(sig, panPos);

        sig = Limiter.ar(sig, 0.6);
        Out.ar(out, sig);
    }).add;

    s.sync;

    ">> S.A.R resonator SynthDef loaded — 6-string harmonic bank.".postln;

    // ── Launch the resonator ─────────────────────────────────
    ~resonator = Synth(\sar_resonator, [
        \activity, 0.0,
        \kp,       1.0,
        \dst,      0.0,
        \events,   0.0,
        \alert,    0.0,
    ]);

    ">> S.A.R resonator running — E2 A2 D3 G3 B3 E4".postln;

    // ── OSC responders ───────────────────────────────────────
    ~resLastPrint = 0;
    ~resCachedActivity = 0.0;

    OSCdef(\sar_res_activity, { |msg|
        var val = msg[1].asFloat;
        ~resCachedActivity = val;
        ~resonator.set(\activity, val);
        if (Main.elapsedTime - ~resLastPrint > 20, {
            (">> RES: activity=" ++ val.round(0.001)).postln;
            ~resLastPrint = Main.elapsedTime;
        });
    }, '/sar/resonator/activity');

    OSCdef(\sar_res_kp, { |msg|
        ~resonator.set(\kp, msg[1].asFloat);
    }, '/sar/resonator/kp');

    OSCdef(\sar_res_dst, { |msg|
        ~resonator.set(\dst, msg[1].asFloat);
    }, '/sar/resonator/dst');

    OSCdef(\sar_res_events, { |msg|
        ~resonator.set(\events, msg[1].asFloat);
    }, '/sar/resonator/events');

    OSCdef(\sar_res_alert, { |msg|
        var val = msg[1].asFloat;
        ~resonator.set(\alert, val);
        if (val > 0.1, {
            (">> RES: ALERT level=" ++ val.round(0.01)).postln;
        });
    }, '/sar/resonator/alert');

    ">> S.A.R resonator OSC responders active.".postln;
});
)
