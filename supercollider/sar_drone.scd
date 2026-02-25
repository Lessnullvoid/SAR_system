// ─────────────────────────────────────────────────────────────
// S.A.R  —  Seismic Activity Drone  (SuperCollider)
// ─────────────────────────────────────────────────────────────
// A GEOLOGICAL drone.  Sensor data shapes the CHARACTER of what
// you hear, not just volume.  Changes happen on a timescale of
// minutes, not seconds.
//
// Silence timer: the drone fades to silence every 9/13/15 min
// (randomly chosen), holds ~80s, then fades back in.
//
// OSC:  /sar/drone/activity, /kp, /dst, /events, /alert
// Standalone:  sclang supercollider/sar_drone.scd
// ─────────────────────────────────────────────────────────────

(
s.waitForBoot({

    SynthDef(\sar_drone, {
        arg out = 0,
            activity = 0.0,
            kp       = 1.0,
            dst      = 0.0,
            events   = 0.0,
            alert    = 0.0;

        var actL, kpL, dstL, evL, alertL;
        var actW, kpW, dstW, evW, alertW;
        var breath1, breath2, breath3, breath4, breath5;
        var tide, tideTrig;
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

        actL   = Lag.kr(activity, 25);
        kpL    = Lag.kr(kp, 30);
        dstL   = Lag.kr(dst, 20);
        evL    = Lag.kr(events, 20);
        alertL = Lag.kr(alert, 15);

        actW   = (actL   + LFNoise2.kr(0.015).range(-0.03, 0.03)).clip(0, 1);
        kpW    = (kpL    + LFNoise2.kr(0.012).range(-0.2, 0.2)).clip(0, 9);
        dstW   = dstL    + LFNoise2.kr(0.018).range(-2, 2);
        evW    = (evL    + LFNoise2.kr(0.01).range(-2, 2)).max(0);
        alertW = (alertL + LFNoise2.kr(0.02).range(-0.02, 0.02)).clip(0, 1);

        breath1 = SinOsc.kr(1/73.0).range(0.82, 1.0);
        breath2 = SinOsc.kr(1/127.0).range(0.85, 1.0);
        breath3 = LFNoise2.kr(1/180.0).range(0.80, 1.0);
        breath4 = SinOsc.kr(1/97.0).range(0.86, 1.0);
        breath5 = LFNoise2.kr(1/240.0).range(0.78, 1.0);

        // Silence timer — triggers every 9, 13, or 15 minutes (random).
        // Envelope: 1→0 (25s) → 0 (80s) → 0→1 (35s) = ~2.3 min episode.
        tideTrig = TDuty.kr(Drand([300, 540, 780], inf), 0, 1.0, 0, 1);
        tide = EnvGen.kr(
            Env([1, 0, 0, 1], [25, 80, 35], [\sin, 0, \sin]),
            tideTrig
        );

        // Report tide transitions
        SendReply.kr(Trig1.kr(tide < 0.01, 1), '/tide_state', [0]);
        SendReply.kr(Trig1.kr(tide > 0.99, 1), '/tide_state', [1]);

        progressRate = evW.linlin(0, 100, 0.00042, 0.0011);
        autoLinear = LFSaw.kr(progressRate, 0).range(0, 6.999);
        autoWander = LFNoise2.kr(0.003).range(-1.0, 1.0);
        autoPhase = (autoLinear + autoWander).mod(7);
        sensorShift = actW.linlin(0, 1, 0, 1.5);
        chordFloat = (autoPhase + sensorShift).mod(7);
        chordIdx = chordFloat.round(1).min(6);

        rootNote    = Lag.kr(Select.kr(chordIdx, #[33, 26, 31, 36, 29, 34, 27]), 20);
        thirdNote   = Lag.kr(Select.kr(chordIdx, #[36, 29, 34, 39, 32, 37, 30]), 20);
        fifthNote   = Lag.kr(Select.kr(chordIdx, #[40, 33, 38, 43, 36, 41, 34]), 20);
        seventhNote = Lag.kr(Select.kr(chordIdx, #[43, 36, 41, 46, 39, 44, 37]), 20);

        rootFreq    = rootNote.midicps;
        thirdFreq   = thirdNote.midicps;
        fifthFreq   = fifthNote.midicps;
        seventhFreq = seventhNote.midicps;

        detuneAmt = dstW.abs.linlin(0, 100, 0.0, 0.012).min(0.012);

        drift1 = LFNoise1.kr(0.04).range(-0.002, 0.002);
        drift2 = LFNoise1.kr(0.03).range(-0.003, 0.003);
        drift3 = LFNoise1.kr(0.05).range(-0.002, 0.002);
        drift4 = LFNoise1.kr(0.025).range(-0.0015, 0.0015);
        drift5 = LFNoise1.kr(0.035).range(-0.002, 0.002);

        vBal1   = LFNoise2.kr(0.008).range(0.75, 1.0);
        vBal2   = LFNoise2.kr(0.006).range(0.55, 1.0);
        vBal3   = LFNoise2.kr(0.009).range(0.45, 1.0);
        vBalSub = LFNoise2.kr(0.005).range(0.70, 1.0);
        vBal7   = LFNoise2.kr(0.007).range(0.60, 1.0);

        v1a = LFSaw.ar(rootFreq * (1 + detuneAmt + drift1), 0, 0.10);
        v1b = LFSaw.ar(rootFreq * (1 - detuneAmt - drift1), 0, 0.10);
        v1 = (v1a + v1b) * vBal1 * breath1;

        v2 = LFTri.ar(
            thirdFreq * (1 + (detuneAmt * 0.5) + drift2 + (v1 * 0.0006)),
            0,
            0.07 * actW.linlin(0.1, 0.35, 0.0, 1.0).clip(0, 1)
        ) * vBal2 * breath2;

        v3 = SinOsc.ar(fifthFreq * (1 + drift3), 0,
            0.05 * actW.linlin(0.3, 0.6, 0.0, 0.85).clip(0, 0.85)
        ) * vBal3 * breath3;

        vSub = SinOsc.ar(rootFreq * 0.5 * (1 + drift4), 0,
            actW.linlin(0, 0.5, 0.10, 0.18).clip(0.10, 0.18)
        ) * vBalSub * breath5;

        v7th = SinOsc.ar(seventhFreq * (1 + drift5), 0,
            (kpW.linlin(4, 7, 0.0, 0.04).max(0)
             * actW.linlin(0.25, 0.5, 0.0, 1.0).clip(0, 1))
        ) * vBal7 * breath4;

        alertNote = rootNote + 13;
        vAlert = SinOsc.ar(alertNote.midicps * (1 + drift2.neg), 0,
            alertW.linlin(0.3, 0.8, 0.0, 0.03).max(0));

        sig = v1 + v2 + v3 + vSub + v7th + vAlert;

        filtFreq = kpW.linexp(0, 9, 80, 1800)
            * LFNoise2.kr(0.012).range(0.82, 1.18);
        filtQ = alertW.linlin(0, 1, 0.45, 0.20);
        sig = RLPF.ar(sig, filtFreq.clip(30, 5000), filtQ);

        grainSig = Dust.ar(evW.linlin(8, 80, 0.2, 5).max(0.1)) * 0.025
            * BPF.ar(WhiteNoise.ar, Lag.kr(thirdFreq * 4, 8), 0.15)
            * actW.linlin(0.2, 0.5, 0.0, 1.0).clip(0, 1);
        sig = sig + grainSig;

        ampBreath = (breath1 * breath2);
        amp = actW.linexp(0, 1, 0.12, 0.28) * ampBreath * tide;
        sig = sig * amp;

        delTime1 = kpW.linlin(0, 9, 0.38, 0.18);
        delTime2 = delTime1 * 0.618;
        delMod = (SinOsc.kr(0.04) * dstW.abs.linlin(0, 100, 0.0, 0.015))
            + LFNoise2.kr(0.015).range(-0.005, 0.005);
        delTime1 = (delTime1 + delMod).clip(0.05, 0.48);
        delFb = actW.linlin(0, 0.6, 0.20, 0.45).clip(0.20, 0.45);
        delSig = AllpassC.ar(sig, 0.5, delTime1, delFb * 3.0);
        delSig = AllpassC.ar(delSig, 0.5, delTime2, delFb * 2.0);
        sig = sig + (delSig * 0.25);

        panPos = LFNoise2.kr(0.008).range(-0.5, 0.5);
        sig = Pan2.ar(sig, panPos);

        revRoom = alertW.linlin(0, 1, 0.50, 0.88)
            * LFNoise2.kr(0.006).range(0.90, 1.0);
        revDamp = kpW.linlin(0, 9, 0.70, 0.30)
            * LFNoise2.kr(0.005).range(0.92, 1.0);
        revMix = actW.linlin(0, 0.5, 0.20, 0.35).clip(0.20, 0.35)
            * breath3
            * tide.linlin(0.0, 0.5, 1.8, 1.0).clip(1.0, 1.8);

        sig = FreeVerb2.ar(sig[0], sig[1], revMix, revRoom.clip(0,1), revDamp.clip(0,1));

        sig = Limiter.ar(sig, 0.7);
        Out.ar(out, sig);
    }).add;

    s.sync;

    ~drone = Synth(\sar_drone, [
        \activity, 0.0, \kp, 1.0, \dst, 0.0,
        \events, 0.0, \alert, 0.0,
    ]);
    ">> Drone running — Am1 → Dm1 → Gm1 → Cm2 → Fm1 → Bbm1 → Ebm1".postln;

    // ── OSC responders with continuous monitoring ──────────
    ~lastPrint = 0;
    ~cachedKp = 1.0;
    ~cachedDst = 0.0;
    ~cachedEvents = 0;
    ~cachedActivity = 0.0;
    ~cachedAlert = 0.0;

    OSCdef(\sar_activity, { |msg|
        var val = msg[1].asFloat;
        ~cachedActivity = val;
        ~drone.set(\activity, val);
        if (Main.elapsedTime - ~lastPrint > 10, {
            (">> DRONE  act=" ++ val.round(0.001)
             ++ "  kp=" ++ ~cachedKp.round(0.1)
             ++ "  dst=" ++ ~cachedDst.round(1)
             ++ "  ev=" ++ ~cachedEvents
             ++ "  alert=" ++ ~cachedAlert.round(0.01)
            ).postln;
            ~lastPrint = Main.elapsedTime;
        });
    }, '/sar/drone/activity');

    OSCdef(\sar_kp, { |msg|
        var val = msg[1].asFloat;
        ~cachedKp = val;
        ~drone.set(\kp, val);
        (">> DRONE  kp=" ++ val.round(0.1)
         ++ (if (val > 4, {" [7th voice active]"}, {""}))
        ).postln;
    }, '/sar/drone/kp');

    OSCdef(\sar_dst, { |msg|
        var val = msg[1].asFloat;
        ~cachedDst = val;
        ~drone.set(\dst, val);
        (">> DRONE  dst=" ++ val.round(1) ++ "nT"
         ++ (if (val.abs > 30, {" [detuning active]"}, {""}))
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
            (">> DRONE  ALERT=" ++ val.round(0.01) ++ " [tension tone]").postln;
        });
    }, '/sar/drone/alert');

    // Tide state monitoring
    OSCdef(\drone_tide, { |msg|
        var state = msg[3].asFloat;
        if (state < 0.5, {
            ">> DRONE  ◼ SILENCE — tide reached zero".postln;
        }, {
            ">> DRONE  ▶ PRESENT — tide back to full".postln;
        });
    }, '/tide_state');

    "".postln;
    ">> S.A.R drone ready — OSC on port 57120".postln;
    ">> Silence timer: every 9/13/15 min, ~2 min episodes".postln;
    "══════════════════════════════════════════════════════".postln;
});
)
