# Using Nomo: a plain-English guide

Nomo helps you run an AI model on a chip using as little energy as possible. It does this by trying many
ways of splitting the model between three styles of computing, and showing you the best trade-offs.

- **Continuous (blue):** normal neural-network layers. Accurate and fast, but they always do all the work.
- **Spiking (amber):** brain-style layers that only send signals when something changes. Much cheaper on
  neuromorphic chips, slightly less accurate, and they need a few time steps per answer.
- **Physics formula (green):** a layer replaced by an exact equation, where one exists (for example, how a
  drone rotates). Exact and very cheap.

Every place where data switches between styles costs a little extra. The design graph marks these with red
badges such as "ANN → SNN".

## 1. Start a search

On the start page there are four steps.

1. **Choose a model.** Pick one of the two examples, or upload your own. Uploads can be ONNX files (the most
   reliable), PyTorch weight files (`.pt` / `.pth`) or a Nomo JSON graph. Nomo lists anything it had to
   guess about your file; read that list.
2. **Choose a chip.** Open "Chip parameters" to enter your chip's real numbers. Until you do, energy and speed
   are estimates from typical values.
3. **Choose a goal.** Battery Saver, Ultra-Low Latency, Balanced Edge or Strict Safety. Each fills in the
   settings in step 4, which you can still change.
4. **Fine-tune (optional).** Set how much accuracy you can give up, switch computing styles on or off, lock
   particular layers to a style or precision, or change how long the search runs. Hover over any dotted
   word for an explanation.

Press **Find the best designs**. The first visit of the day can take a minute while the free server wakes up.

## 2. Read the results

- **The 3D plot** shows every design tried. Dark dots are the best trade-offs; pink ones break a limit you set.
  Use the sliders above it to hide designs that are too slow, too power-hungry or not accurate enough. Click a
  dot to look at that design.
- **The right-hand panel** shows the selected design's energy, response time and accuracy, compared with
  running everything as standard layers, plus a one-paragraph explanation.
- **The design graph** at the bottom shows each layer in its style. Click a layer to see its memory, energy
  and cores, to ask why it was chosen, or to **lock it and search again**.

## 3. Ask Copilot

Press **Ask Copilot** and ask things like "Why is fc1 spiking?", "How do I cut energy by another 20%?" or
"Explain the trade-offs". Answers are worked out from your search. For example, it really re-tests the design
with that layer changed. Buttons under an answer show a design or start a new search for you.

## 4. Export

When the search has finished, press **Export**, tick the formats you want, and download the zip. If a format
can't be made for this design, it's greyed out with the reason. The zip contains a README explaining every
file. Start with `report.pdf`.

## What to keep in mind

- Chip numbers are estimates until you enter real ones. Say so when quoting results.
- Accuracy is estimated, not measured. Check the exported model on your own test data.
- The example models use untrained demo weights, so their exports show the structure but won't give useful
  predictions. Upload your own model for real results.
- Free hosting forgets searches and uploads when it restarts or sleeps.
