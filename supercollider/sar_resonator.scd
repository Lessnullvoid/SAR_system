// ─────────────────────────────────────────────────────────────
// S.A.R  —  Sympathetic String Resonator  (SuperCollider)
// ─────────────────────────────────────────────────────────────
// 6-string harmonic bank (E2 A2 D3 G3 B3 E4).
// Sensor data controls which strings vibrate and how rich
// their partials are.
//
// Silence timer: fades to silence every 3/7/11 min (random),
// holds ~50s, then fades back in.
//
// OSC:  /sar/resonator/activity, /kp, /dst, /events, /alert
// Standalone:  sclang supercollider/sar_resonator.scd
// ─────────────────────────────────────────────────────────────

(
s.waitForBoot({

    SynthDef(\sar_resonator, {
        arg out = 0,
            activity = 0.0,
            kp       = 1.0,
            dst      = 0.0,
            events   = 0.0,
            alert    = 0.0;

        var actL, kpL, dstL, evL, alertL;
        var actW, kpW, dstW, evW, alertW;
        var breath1, breath2;
        var tide, tideTrig;
        var stringFreqs;
        var stringGates, partialDepth;
        var dstDetune;
        var sig, impulseSig, impulseRate;
        var dissonantSig, disFreq1, disFreq2;
        var masterAmp;
        var panPos;

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

        breath1 = SinOsc.kr(1/83.0).range(0.80, 1.0);
        breath2 = LFNoise2.kr(1/150.0).range(0.82, 1.0);

        // Silence timer — triggers every 3, 7, or 11 minutes (random).
        // Envelope: 1→0 (15s) → 0 (50s) → 0→1 (20s) = ~1.4 min episode.
        tideTrig = TDuty.kr(Drand([180, 420, 660], inf), 0, 1.0, 0, 1);
        tide = EnvGen.kr(
            Env([1, 0, 0, 1], [15, 50, 20], [\sin, 0, \sin]),
            tideTrig
        );

        // Report tide transitions
        SendReply.kr(Trig1.kr(tide < 0.01, 1), '/res_tide_state', [0]);
        SendReply.kr(Trig1.kr(tide > 0.99, 1), '/res_tide_state', [1]);

        stringFreqs = #[82.41, 110.0, 146.83, 196.0, 246.94, 329.63];

        stringGates = [
            1.0,
            1.0,
            actW.linlin(0.15, 0.35, 0.0, 1.0).clip(0, 1),
            actW.linlin(0.35, 0.55, 0.0, 1.0).clip(0, 1),
            actW.linlin(0.55, 0.75, 0.0, 1.0).clip(0, 1),
            actW.linlin(0.75, 0.90, 0.0, 1.0).clip(0, 1),
        ];

        partialDepth = kpW.linlin(0, 9, 2, 8).round(1).clip(1, 8);
        dstDetune = dstW.abs.linlin(0, 80, 0.0, 0.008).min(0.008);
        impulseRate = evW.linlin(0, 60, 0.08, 2.5).clip(0.05, 3.0);

        sig = DC.ar(0);

        // String 0 — E2 82.41 Hz (always on)
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

        // String 1 — A2 110 Hz (always on)
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

        // String 2 — D3 146.83 Hz
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

        // String 3 — G3 196 Hz
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

        // String 4 — B3 246.94 Hz
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

        // String 5 — E4 329.63 Hz
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

        // Impulse excitation — sparse resonant plucks
        impulseSig = DC.ar(0);
        6.do({ |i|
            var trig = Dust.ar(impulseRate * stringGates[i] * LFNoise2.kr(0.01).range(0.3, 1.0));
            var burst = trig * WhiteNoise.ar(0.15);
            var reson = BPF.ar(burst, stringFreqs[i], 0.005);
            impulseSig = impulseSig + reson;
        });
        sig = sig + (impulseSig * evW.linlin(0, 30, 0.3, 1.0).clip(0.2, 1.0));

        // Dissonant partials — alert tension (7th + 11th harmonics of E2)
        disFreq1 = stringFreqs[0] * 7 * (1 + LFNoise1.kr(0.02).range(-0.003, 0.003));
        disFreq2 = stringFreqs[0] * 11 * (1 + LFNoise1.kr(0.015).range(-0.002, 0.002));
        dissonantSig = SinOsc.ar(disFreq1, 0, 0.012) + SinOsc.ar(disFreq2, 0, 0.008);
        dissonantSig = dissonantSig * alertW.linlin(0.25, 0.7, 0.0, 1.0).clip(0, 1);
        sig = sig + dissonantSig;

        // Output shaping
        masterAmp = actW.linexp(0, 1, 0.15, 0.40) * breath1 * breath2 * tide;
        sig = sig * masterAmp;
        sig = HPF.ar(sig, 60);

        // Widen spatial field as resonator retreats
        panPos = LFNoise2.kr(0.005).range(-0.3, 0.3)
            * tide.linlin(0.0, 0.4, 2.0, 1.0).clip(1.0, 2.0);
        sig = Pan2.ar(sig, panPos.clip(-1, 1));

        sig = Limiter.ar(sig, 0.6);
        Out.ar(out, sig);
    }).add;

    s.sync;

    ~resonator = Synth(\sar_resonator, [
        \activity, 0.0, \kp, 1.0, \dst, 0.0,
        \events, 0.0, \alert, 0.0,
    ]);
    ">> Resonator running — E2 A2 D3 G3 B3 E4".postln;

    // ── OSC responders with continuous monitoring ──────────
    ~resLastPrint = 0;
    ~resCachedActivity = 0.0;
    ~resCachedKp = 1.0;
    ~resCachedDst = 0.0;
    ~resCachedEvents = 0;
    ~resCachedAlert = 0.0;

    OSCdef(\sar_res_activity, { |msg|
        var val = msg[1].asFloat;
        ~resCachedActivity = val;
        ~resonator.set(\activity, val);
        if (Main.elapsedTime - ~resLastPrint > 10, {
            (">> RES  act=" ++ val.round(0.001)
             ++ "  kp=" ++ ~resCachedKp.round(0.1)
             ++ "  dst=" ++ ~resCachedDst.round(1)
             ++ "  ev=" ++ ~resCachedEvents
             ++ "  alert=" ++ ~resCachedAlert.round(0.01)
             ++ "  strings=" ++ [
                 1, 1,
                 if(val > 0.15, {1}, {0}),
                 if(val > 0.35, {1}, {0}),
                 if(val > 0.55, {1}, {0}),
                 if(val > 0.75, {1}, {0}),
             ].sum
            ).postln;
            ~resLastPrint = Main.elapsedTime;
        });
    }, '/sar/resonator/activity');

    OSCdef(\sar_res_kp, { |msg|
        var val = msg[1].asFloat;
        ~resCachedKp = val;
        ~resonator.set(\kp, val);
        (">> RES  kp=" ++ val.round(0.1)
         ++ "  partials=" ++ val.linlin(0, 9, 2, 8).round(1)
        ).postln;
    }, '/sar/resonator/kp');

    OSCdef(\sar_res_dst, { |msg|
        var val = msg[1].asFloat;
        ~resCachedDst = val;
        ~resonator.set(\dst, val);
        (">> RES  dst=" ++ val.round(1) ++ "nT"
         ++ (if (val.abs > 20, {" [beating active]"}, {""}))
        ).postln;
    }, '/sar/resonator/dst');

    OSCdef(\sar_res_events, { |msg|
        var val = msg[1].asFloat;
        ~resCachedEvents = val;
        ~resonator.set(\events, val);
    }, '/sar/resonator/events');

    OSCdef(\sar_res_alert, { |msg|
        var val = msg[1].asFloat;
        ~resCachedAlert = val;
        ~resonator.set(\alert, val);
        if (val > 0.1, {
            (">> RES  ALERT=" ++ val.round(0.01) ++ " [dissonant partials]").postln;
        });
    }, '/sar/resonator/alert');

    // Tide state monitoring
    OSCdef(\res_tide, { |msg|
        var state = msg[3].asFloat;
        if (state < 0.5, {
            ">> RES  ◼ SILENCE — tide reached zero".postln;
        }, {
            ">> RES  ▶ PRESENT — tide back to full".postln;
        });
    }, '/res_tide_state');

    OSCdef(\sar_shutdown, {
        ">> S.A.R shutdown received — freeing synth…".postln;
        ~resonator.free;
        { 0.5.wait; ">> sclang exiting.".postln; 0.exit; }.fork;
    }, '/sar/shutdown');

    "".postln;
    ">> S.A.R resonator ready — OSC on port 57120".postln;
    ">> Silence timer: every 3/7/11 min, ~1.4 min episodes".postln;
    "══════════════════════════════════════════════════════".postln;
});
)
