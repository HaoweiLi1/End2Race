# PPO V1.1

V1.1 completed 20 updates and 512,000 transitions from a fresh BC start. Zero-LR and nonzero smoke gates passed before the pilot; frozen actor groups and `log_std` remained unchanged.

The selected checkpoint was update 2: 15 ego collisions, 232 follow, and 353 overtake on the 600-case ego-scope panel. Update 20 produced 19/225/356 and is retained as the final-update comparison model.

The pilot did not prove stable collision improvement across training; the checkpoint selection is therefore the recorded update-2 result, not the final update.
