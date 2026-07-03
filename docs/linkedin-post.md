I taught an AI to recognize individual humpback whales it had never seen before — from a photo of the tail.

Then I connected it to shipping-traffic data so it can flag which whales are swimming into the path of the next vessel.

Here's the honest story of how it went from 2.5% to 62% accuracy — and the one habit that got it there.

---

The problem: given a photo of a humpback's fluke, name the individual out of a catalog of ~24,000 whales. The marks that tell them apart are tiny — notches, scars, pigment on the trailing edge. And the whale you're identifying was never in the training set. That's the real-world version of the task: not "re-spot a whale you memorized," but "recognize one you've never seen."

I built the measuring rig before I built a single model. Split by whole individual so no whale leaks across train/test. Score against a disjoint gallery. Exact search, not the fast approximate index. Spend the test set exactly once. If you can't trust the number, improving it is theater.

The journey, blow by blow:

→ Raw CLIP, no training: 2.5%. The floor. It knows "whale," not *which* whale.
→ First fine-tune with triplet loss: it collapsed — every whale mapped to the same point.
→ Switched the loss function, scaled up: 27%. A real lift... and then a plateau.

And this is the part I'm actually proud of: I refused to ship 27%. "Improved" and "good" are different questions.

So instead of throwing a bigger model at it, I ran two 10-minute experiments first.

Experiment 1 said: cropping to the fluke won't help — because when I actually *looked* at the photos, they were already tight close-ups. The entire "crop first" playbook everyone repeats was solving a problem I didn't have.

Experiment 2 said: resolution is the lever, model size is not. Identity lives in fine detail that low resolution blurs away — and a bigger model can't recover detail the input already threw away.

Both experiments were right. Swapping to the backbone they pointed at landed the model at 62% top-1 accuracy on whales it had never seen — about 25x the original — trained in roughly an hour on a laptop.

The transferable lesson isn't about whales. It's this:

Attack the levers top-down — what the model SEES before how it's TAUGHT before how BIG it is. And look at your own data before you import someone else's playbook. Four eyeballed images overturned an entire assumed workstream.

I'm not claiming a world record. The teams that win these competitions use ensembles of large models on multi-GPU clusters. What I can do is name the exact gap to world-class and tell you what it costs — which, honestly, is the more useful skill.

Now the model plugs into an agent that chains it across sighting and shipping data to answer the question that actually matters for conservation: *is this animal at risk?*

Happy to share the write-up and code if it's useful to anyone working on fine-grained re-identification, marine conservation tech, or agentic systems on top of real ML.

What would you have probed first?

#MachineLearning #AI #Conservation #ComputerVision #AgenticAI
