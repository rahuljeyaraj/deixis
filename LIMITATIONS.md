# What `deixis` cannot do (yet)

`deixis` turns words like "here", "that one" and "this big", plus pointing, into a place a robot can act on. This page lists what it does **not** do, and why. Numbers come with their source. "To measure" marks numbers we will fill in from our own tests.

Draft of 4 Oct 2026, written with the plan (`docs/plan.md`). Keep it true: when a limit is lifted, move it to `CHANGELOG.md`.

---

## L1. One flat surface, in 2D

Every place is in **one flat frame per robot**: a sheet of paper, a floor, a table top. Positions are x and y in millimetres. Height only appears in one case: "put that on that" gives the height of the thing underneath (`z_mm`), if the robot told us how tall it is.

- **So:** no "fly up to here", no "this high", no shelves at several heights, no curved or sloping surfaces.
- **Why:** the hand trackers and the timing rules work on a plane. Height gestures need a side view or 3D body tracking.
- **Later:** height on samples and holds (plan task C24).

## L2. No pointing at far-away places

`deixis` works when the pointer is **at** the place: a fingertip on or just above paper, feet standing on the spot, a mouse click, a laser spot, a gaze point. It does not follow a ray from your arm across a room.

- **Number:** pointing at a wall 2 to 3 m away is off by **61.3 cm on average**, even with the best ray model (eye through fingertip); a learned correction cut that by 37.3 % (Mayer, Wolf, Schneegass and Henze, CHI 2015).
- **Why:** the error grows with distance, and a ray needs 3D body tracking that our helpers don't have.
- **Instead:** walk to the spot and say "here" (feet), or use a laser pointer.
- **Later:** cutting a 3D ray with the floor, with an uncertainty that grows with distance (plan task C23).

## L3. Precision is the pointer's, not ours

`deixis` passes on the uncertainty it is given (the hold's spread plus a base value per robot) and reports it as a covariance. How exact a place is depends on the camera and the setup:

| Setup | Number | Source |
|---|---|---|
| Pen plotter, finger on the paper, real rig | 0.6 mm median placement error | our measurements, Oct 2026 (presentation research, section 3.4) |
| Virtual plotter, finger touching | 0.14 mm median, 1.04 mm worst (80 runs) | `deixis-doodle/docs/sessions/virtual-report.md` |
| Virtual plotter, finger 8 mm above the paper | 3.2 mm median | same |
| Real plotter camera, parallax | a touching fingertip reads about 3 mm toward the user, a hovering one about 7 mm | `deixis-doodle` plan, weak point W51 |
| Web demo, angled view of a 600 × 400 mm table | 0.6 to 0.7 mm per screen pixel near you, about 1.7 mm per pixel (depth) at the far edge; 3 px of fingertip jitter ≈ 5 mm at the far edge | our spike, 4 Oct 2026 |
| Vacuum, feet on the floor from a high camera | to measure | |

## L4. Settings are tuned per robot, on one person

A hold ("the pointer stayed still") is defined by speeds and times in the robot's profile: under 20 mm/s for 0.15 s for a finger over paper; much slower and wider limits for feet. These numbers are starting points from the literature, tuned on **one person** (the builder) per robot.

- **So:** a new robot needs its own profile, and new people may need different values (for example someone with a tremor).
- **Why:** a finger over paper and feet in a room move very differently. There is no single right number.

## L5. The timing rule assumes the gesture comes first

For each pointing word, `deixis` looks for a hold from **1.5 s before** the word starts to **0.5 s after** it ends. Without a pointing word, it takes the last hold from 2 s before speech to its end.

- **Number:** when people pointed and spoke one after the other, the pointing came first **99 % of the time**, with lags of one to two seconds (Oviatt, DeAngeli and Kuhn, CHI 1997).
- **So:** a person who points much earlier, or well after the word, is not matched. The robot then **asks** instead of guessing.

## L6. The scene must be given

`deixis` does not see. The robot side must tell it what things are there (their boxes, labels, colours, heights). On the plotter, the robot's own drawings are known exactly and the person's marks come from ink segmentation, with names from a vision model (Cosmos). In the web demo, the blocks are known.

- **So:** "that one" only finds things the robot side reported.

## L7. English only

The pointing words ("here", "there", "this", "that", "these", "those", "so") and the size words are English. The words live in the robot's profile, so another language is a list of words, not new code, but no other language has been tried.

## L8. Size is one number

"This big" sets the **larger side** of the thing's box. It does not set the other side, the height or the shape.

- **Number:** an overhead camera sees hands held 15 cm above the paper about **30 % bigger** (camera 60 cm up). The correction belongs to the camera side.
- **Number:** a one-hand pinch reaches about 15 to 20 cm, two hands about 1 m (design estimate, not measured).
- **Note:** we found **no published study** of how accurately people show a size with their hands (literature review, section 3.6). Our own numbers are a first measurement, not a norm.

## L9. One point of view for "left", "above" and "next to"

Relations are worked out in **one viewer's frame**, set in the profile (for example "the paper as the person sees it": +x to their right, +y away from them). That frame must line up with the frame's axes.

- **So:** no "in front of the car" meaning the car's own front (an intrinsic frame), and no relations for a person who turns around while speaking.
- **Why:** vision-language models are known to be weak and inconsistent about whose "left" is meant (COMFORT, Zhang et al., ICLR 2025), so `deixis` fixes the frame in code.

## L10. One person at a time

`deixis` does not know who spoke or whose hand it is. Two people pointing at once will confuse it.

## L11. Words and pointing need one clock

Word times and pointer times must come from the same clock, to within a few tens of milliseconds. `deixis` trusts the timestamps it gets.

- **Number:** the speech model's word times are good to about 50 ms (presentation research, section 3.3).

## L12. Not a safety system

`deixis` never checks reach, collisions, or whether a person is in the way. A place outside the work area is flagged or asked about, **never moved**. Stopping, the "hands clear" check and waiting for a person to step out of a box are the robot's job.

## L13. Not yet: direction, rotation, traced shapes, many things

- "Go **that way**" (a direction), "turn it **this way**" (a rotation) and "move **like this**" (a traced shape) have no message type yet.
- "**These three**" and circling a group are not supported (only "between **these two**").
- Planned as `Where` version 1.1 (plan tasks C25, C26).

## L14. One frame per request

All places in one request are in one frame. Two robots in one room work if they share that frame (one camera, one floor), and each robot maps the frame to its own map. Two robots with different frames in one sentence need two requests for now (plan task C27).

## L15. The MCP server is local and per sentence

The MCP server runs on the same computer as the robot app (stdio) and is called **once per sentence**, not once per camera frame. Pointer samples are sent in a batch with the sentence.

- **Number:** time per call: to measure (target under 30 ms for 900 samples).
- **Later:** a networked server that outside agents can query live (plan task C22).

---

## Sources

- Mayer, S., Wolf, K., Schneegass, S. and Henze, N. "Modeling distant pointing for compensating systematic displacements." CHI 2015.
- Oviatt, S., DeAngeli, A. and Kuhn, K. "Integration and synchronization of input modes during multimodal human-computer interaction." CHI 1997.
- Zhang et al. "Do vision-language models represent space and how?" (COMFORT). ICLR 2025. https://arxiv.org/abs/2410.17385
- Our literature review: `deixis-doodle/docs/robot-deixis-literature-review.md`, sections 3.2 to 3.6.
- Our measurements: `deixis-doodle/docs/sessions/virtual-report.md`; `deixis-doodle/docs/point-and-draw-plan.md` (section 4.3 and weak point W51); the angled-view spike, `deixis-doodle-scratch/core/spikes/angled_homography.py` (4 Oct 2026).
