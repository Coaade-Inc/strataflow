# StrataFlow

## Frontier AI. In your pocket. No GPU. No cloud.

**A Coaade Inc. investor brief.**

---

## The impossible thing we already do

A **19 GB frontier AI model** runs **end to end on a 12 GB laptop-class machine
with no GPU** - and produces output that is **byte-for-byte identical** to the
industry reference.

Read that again. The model is bigger than the memory. It runs anyway. Perfectly.

By the rule the entire industry treats as a law of physics - *the model must fit
in memory, memory means a GPU, a GPU means the cloud* - this is not supposed to
be possible. We did it. It's shipped, it's reproducible, and the memory it uses
is **tunable from ~2.2 GB to ~5.5 GB with a single dial.**

That is the whole company in one sentence: **we turned memory from a wall into a
dial.**

---

## Why everyone else is stuck

Running a big model today means one of two bad deals:

- **Rent the cloud.** Your data leaves your device. You pay per token, forever.
  When their servers blink, you're dark.
- **Buy the GPUs.** Thousands of dollars, heat, power - and the model *still* has
  to cram into VRAM.

Both lock the most capable models out of the device in your hand, the laptop on
your desk, and the robot at the edge. "Big model = big GPU = someone else's
datacenter" has been accepted as gravity.

**We broke gravity.**

A Mixture-of-Experts model is enormous on disk, but for any single token only a
sliver of it is actually used. StrataFlow keeps that sliver in RAM and streams
the rest from the SSD the device already has. Memory stops deciding *whether* a
model runs. It only decides *how fast.* **More memory buys speed - never
permission.**

---

## What this unlocks

- **Personal AI.** A frontier model running *on your own laptop or phone.* Fully
  private. Fully offline. Zero per-token cost. Nothing leaves your device - ever.
- **Physical AI.** Robots, drones, vehicles, appliances with large-model
  reasoning at the edge, where the datacenter doesn't reach, bandwidth is thin,
  and every watt counts. Streaming from storage is the *perfect* shape for a
  device with a big SSD and small memory.
- **Enterprise, on your terms.** The same engine scales *up* onto GPUs when they
  exist - on hardware the organization controls, never shipping data to anyone.

One engine. From a phone to a GPU server.

---

## What we've already proven

Every claim here is backed by shipped, reference-gated, four-platform-tested code.
We don't demo; we prove.

### The headline, on a deliberately weak machine

On a free cloud CPU box - **12 GB RAM, 2 cores, an ordinary SSD, no GPU**,
*weaker* than the MacBooks and laptops we actually target:

- **A real 47-billion-parameter Mixtral model, ~19 GB on disk, runs end to end.**
- Output **byte-identical** to the reference. We don't approximate the model to
  make it fit - **we run the real thing, exactly.**
- Resident weights ~0.9 GB; whole-process memory tunable **2.2 GB to 5.5 GB.**

### We own the engine

StrataFlow runs its **own inference forward pass.** We use the industry-standard
stack only as a tokenizer and as our correctness oracle - it is *out* of the
inference path. The streaming, the memory tiering, the scheduling: all ours. No
black box we're trapped behind. Every efficiency the field invents, we can adopt.

### One dial for the whole fit-vs-speed tradeoff

- **`auto`** - the good-neighbor default. Leaves headroom so the rest of your
  apps keep running.
- **`max`** - dedicate the machine for maximum speed.
- **`target`** - name your memory ceiling; the engine fits inside it, or tells you
  the exact honest minimum.

Under the hood, one dial silently coordinates a stack of memory-reduction
techniques - expert streaming, automatic per-model tuning, trunk streaming,
lossless attention-cache paging, honest pre-flight budgeting, and
compress-what-you-stream. **Every one is lossless: the answer is identical at
every memory setting.** The dial changes cost and speed, never the output.

### We run the models that actually exist

Quantized weights (how real models ship) *and* full precision. Real models
downloaded from the open ecosystem, not just lab fixtures. Mixture-of-Experts and
dense architectures today, more coming.

### The discipline that makes it investable

This isn't a tape-and-hope demo. **Every single change** is gated byte-identical
against the reference, compiles clean under two compilers with warnings-as-errors,
and passes CI on **four platforms** (Linux x2, macOS, Windows) before it merges.
For a product whose promise is *"exactly correct, just radically cheaper to run,"*
that provable correctness **is the moat.** Anyone can make a model *almost* run
small. Making it run *exactly right*, provably, everywhere, is the hard part - and
it's done.

---

## The proof, in numbers

> The table lands the headline at a glance. Figures marked **[RESULT PENDING]**
> are reserved for the latest hardware run and will be filled from measured data.

| What | Result |
|---|---|
| Model | Mixtral-8x7B (47B params, MoE), ~19 GB on disk |
| Machine | 12 GB RAM, 2 cores, SSD, **no GPU** |
| Runs end to end | **Yes** |
| Correctness vs. reference | **Byte-identical** |
| Resident weights | ~0.9 GB |
| Memory, tunable | **~2.2 GB -> ~5.5 GB** via one dial |
| Throughput (weak 2-core floor) | **[RESULT PENDING]** |
| Throughput (target laptop / Mac) | **[RESULT PENDING]** |
| Throughput (with GPU) | **[RESULT PENDING]** |
| Peak memory, target mode | **[RESULT PENDING]** |

---

## Why StrataFlow wins

1. **We break the memory-capability law.** Competitors fit models into memory. We
   make memory irrelevant to *whether* a model runs - relevant only to *how fast.*
   That's a category difference, not an increment.
2. **We own the whole engine.** Forward pass, streaming, tiering, scheduling - all
   ours. Every gain in the field is ours to adopt.
3. **We are provably correct.** Byte-identical to reference, gated on every merge,
   across four platforms. In a field of "good-enough" approximations, we run the
   *real* model, *exactly* - and prove it on demand.
4. **One engine, phone to datacenter.** From a 2 GB footprint on a laptop to
   multi-GPU on-prem, governed by one dial. One product, the whole range.
5. **CPU-first, GPU-optional.** The industry assumes the GPU. We assume the device
   you already own and treat the GPU as a bonus. That inverts the cost of running
   large models.

---

## Where this goes

> **This section is forward-looking. The items below are goals and projections,
> not measured results.** They state the direction and ambition of the roadmap;
> the measured facts are in the sections above.

- **Speed that disappears as a concern.** We've proven *fit*; next we make it
  *fast* on everyday hardware - GPU offload, overlapped I/O, smarter caching, and
  speculative decoding. *(Goal.)*
- **Frontier models in every pocket.** The direction: the biggest open models
  running interactively on a phone and a laptop, fully private. *(Vision.)*
- **Physical AI as a default.** Large-model reasoning as a standard capability on
  edge devices and robots, not an exotic one. *(Vision.)*
- **Enterprise multi-GPU, streamlined.** The same engine scaling cleanly across
  many GPUs on-prem, with the GPU an accelerator and never a requirement. *(Roadmap.)*
- **Any model, any format, any machine.** Full breadth of architectures and weight
  formats, on whatever hardware the user already has. *(Roadmap.)*

---

## The ask

StrataFlow has proven the hard, "impossible" part: **a frontier model running
correctly and exactly on hardware that cannot, by conventional rules, hold it.**
What remains is execution - speed, breadth, and reach - not invention.

We're putting frontier AI in everyone's pocket. **[RESULT PENDING: round / use of
funds / milestones.]**

---

*StrataFlow is a source-available project of Coaade Inc. Proven capabilities are
backed by merged, reference-gated, multi-platform-tested pull requests and are
reproducible on commodity hardware. The "Where this goes" section is explicitly
forward-looking: goals and projections, not results. `[RESULT PENDING]` fields
are reserved for measured data supplied separately.*
