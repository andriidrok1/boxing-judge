Was inspired by Ammar Abu-Qalbain's post where he built a small machine learning model that scores MMA fights. I like boxing more, so I built my own and pointed it at Fury vs Usyk 1.

What it does: YOLO11 pose on every frame of the broadcast, camera-cut detection so tracks don't bleed across replays, CLIP zero-shot to tell the fighters from the referee (a text prompt, no training), trunks colour to tell Fury from Usyk, then hand-written judging logic on top: punches thrown, ring control, aggression, 10-point must.

Result: 116-111 Usyk. The judges had it 115-112 Usyk, 114-113 Usyk, 114-113 Fury. Same winner, one round wider than the widest card. It agrees with the judges on 6 to 9 of the 12 rounds, and where it disagrees is exactly where they disagreed with each other.

The part I'm most careful about: it counts punches thrown, not landed. At 25 fps a jab is three frames of motion blur, and whether it landed or hit a glove is often invisible from one camera. So anything a fighter won on accuracy looks like a coin flip to it. I'd rather report thrown honestly than landed badly.

Ground truth is external only: the three official cards, CompuBox totals (system: Fury 450 / Usyk 435 thrown, CompuBox: 496 / 407), and one round I checked frame by frame, 235 candidate punches viewed as 5-frame strips. Precision at the threshold I ship is 0.79. That was the lesson from Ammar's post and I kept it from day one: never label your data with your own tracker.

Bugs worth sharing:

The referee. Under the red Riyadh lighting his white shirt has the same HSV as skin, so every colour rule called him a fighter. And after the knockdown the camera holds on Fury's bare chest for six seconds and CLIP called that crop "referee". Fix was embarrassing: a referee wears a shirt, so a bare torso can never be the ref, whatever the model says.

The swap. In a clinch, YOLO's box for Fury widens until it reaches where Usyk's track was a few frames ago, and the tracker hands Fury's box to Usyk's track. Position alone can't tell them apart there. I threw out the tracker, link detections myself with a joint per-frame assignment, and let colour and CLIP veto a link.

The phantom punch. When the wrist keypoint drops out for a few frames, YOLO parks it near the body, then it snaps back onto the extended arm. To a velocity detector that snap is a punch. One punch became three.

And the one a friend caught watching the clip: "Fury barely throws and he's already at 5". Two of those were his arm dropping after a break. My scale normalisation switched estimators whenever the hips left the frame, and a 3x scale jump looks exactly like a fast punch.

For the demo clip I stopped trusting the detector: every candidate in those 30 seconds viewed frame by frame, 12 real punches, each point in the overlay comes with a photo of the punch it was given for. The raw detector finds 8 of the 12 and adds 3 that aren't punches. That gap is the honest number for where this is.

Next: landed vs blocked with a higher-fps source, and a second fight to see how many of my thresholds are really Fury-Usyk thresholds.

Code: github.com/andriidrok1/boxing-judge
