<div align="center">
  <img src="logo.png" alt="Nora logo" width="220">
  <h1>Nora</h1>
  <p><strong>A living, continually learning, biologically inspired cognitive architecture.</strong></p>
  <p>Sparse memory. Grounded reasoning. A persistent inner state. A world in which actions have consequences.</p>
</div>

---

> **Project status:** Nora is research software, not a finished artificial general intelligence. The repository contains working cognitive subsystems, targeted ablations, and reproducible evidence. It also records failed experiments and unresolved thresholds. Read [Evidence and limits](#evidence-and-limits) before treating any result as a general capability claim.

## Contents

- [What Nora is](#what-nora-is)
- [The design thesis](#the-design-thesis)
- [Architecture](#architecture)
- [The cognitive subsystems](#the-cognitive-subsystems)
- [Learning and memory](#learning-and-memory)
- [Language, reasoning, and action](#language-reasoning-and-action)
- [Self-model, life, and deliberation](#self-model-life-and-deliberation)
- [Running Nora](#running-nora)
- [Repository guide](#repository-guide)
- [Evaluation and reproducibility](#evaluation-and-reproducibility)
- [Evidence and limits](#evidence-and-limits)
- [The road ahead](#the-road-ahead)
- [Contributing](#contributing)
- [Collaborators](#collaborators)
- [License and further reading](#license-and-further-reading)

## What Nora is

Nora is an experimental cognitive architecture for an agent that **learns while it lives and acts**. It is not organized around a single prompt-response model. The system combines:

- a simulated physical world with resources, hazards, motion, and metabolism;
- a persistent body and homeostatic state;
- sparse, biologically inspired learning based on the mushroom-body principle;
- a global workspace and a typed cognitive bus for competition and broadcast;
- learned prediction, planning, causal models, and counterfactual evaluation;
- episodic, semantic, active, passive, and disk-backed memory;
- a Turkish language stack with separate comprehension and articulation responsibilities;
- a self-model that tracks values, abilities, continuity, mortality, and self-beliefs;
- explicit test harnesses for intervention, ablation, retention, and held-out generalization.

The word **living** has an operational meaning here. Nora has ongoing needs, a changing internal state, a history, and a mortality mechanism. It does not mean that the repository contains biological life. Likewise, a first-person self-ascription is a behavior of the self-model, not an independent measurement of phenomenal consciousness.

Nora is local-first. The core runtime does not require a hosted model API or an API key. Optional research and corpus-ingestion paths can be enabled explicitly.

## The design thesis

Most conversational systems begin with a text interface and add memory or tools around it. Nora begins with a different question:

> What would an artificial mind need in order to learn from consequences, preserve an identity across time, and explain what changed its decision?

The repository explores that question through executable mechanisms rather than a single aggregate score.

### Design commitments

1. **Learning is continuous.** New information should be usable after one presentation. Epochs and replay are not accepted as the main explanation for first-learning success.
2. **Generalization is tested, not assumed.** Paraphrase, new contexts, held-out relations, and unseen combinations matter more than recalling a training sentence.
3. **Retention is a first-class constraint.** New learning must not silently destroy earlier knowledge. Negative retention results remain visible.
4. **The mushroom-body principle is functional.** Sparse Kenyon-cell-like codes, fast local updates, gating, stiffness, and bounded resources are used because they can be measured—not as anatomical decoration.
5. **Meaning is not delegated to the articulator.** Neural language components can propose or articulate; semantic claims must be grounded in the graph, HDC/MB state, or an explicit verification path.
6. **Intervention is stronger than description.** A causal claim should survive an actual `do(X)` intervention, an ablation, or a counterfactual rerun.
7. **Uncertainty has structure.** Unknown relations, provisional discoveries, missing evidence, and inaccessible memory are different states.
8. **Resource use is part of the research question.** Disk-backed knowledge and bounded active computation are preferred over allowing every new experience to expand the working set without limit.
9. **Evidence is allowed to be inconvenient.** A failed arm, a seed-dependent result, or an unmet threshold is recorded rather than renamed as success.

## Architecture

At a high level, a Nora run is a loop between body, world, learning, deliberation, and action:

```text
                         ┌──────────────────────────────┐
                         │       Simulated world        │
                         │ physics · resources · hazards│
                         └──────────────┬───────────────┘
                                        │ observation
                                        ▼
┌──────────────┐              ┌──────────────────────────────┐
│ Body / life  │─────────────▶│ Perception and interoception │
│ needs · age  │              │ vision · touch · vitals      │
│ mortality    │              └──────────────┬───────────────┘
└──────┬───────┘                             │
       │                                     ▼
       │                         ┌──────────────────────────────┐
       │                         │ Sparse learning and memory   │
       │                         │ MB / HDC · Graph · episodes  │
       │                         └──────────────┬───────────────┘
       │                                    candidates
       │                                     ▼
       │                         ┌──────────────────────────────┐
       │                         │ Workspace and cognitive bus  │
       │                         │ competition · broadcast · WM │
       │                         └──────────────┬───────────────┘
       │                                     ▼
       │                         ┌──────────────────────────────┐
       │                         │ Deliberation and self-model  │
       │                         │ predict · compare · intervene│
       │                         │ value · volition · self-belief│
       │                         └──────────────┬───────────────┘
       │                                     ▼
       │                         ┌──────────────────────────────┐
       │                         │ Action / language articulation│
       │                         │ movement · speech · explanation│
       │                         └──────────────┬───────────────┘
       │                                        │
       └──────────── consequences ◀─────────────┘
```

The diagram is a responsibility map, not a claim that every module is a neural network or that every module has the same level of maturity. Some components are learned, some are symbolic, and some are deliberately explicit experimental instruments.

## The cognitive subsystems

### World, body, and homeostasis

`nora_world.py` supplies the environment in which behavior has costs. Depending on the experiment, it includes Newtonian motion, friction, momentum-preserving collisions, movable obstacles, day/night changes, wind, metabolism, food, water, hazards, and rule changes such as inverted controls or poisoned food.

The agent does not receive a convenient coordinate readout as a substitute for perception. The world and mind experiments use egocentric sensing, interoception, touch, proprioception, and a compass. Location confidence and uncertainty are therefore part of the problem.

The living-mind model tracks needs and derives value from bodily consequences. In the original world-model formulation, homeostatic error is expressed as:

```text
H = Σ(1 − need)²
```

Changes in this error contribute to valence and slower mood dynamics. Reward is consequently tied to the state of the agent rather than being only an external text label.

`nora_nervous.py` and the mortality path add bodily signaling, pain-related responses, and irreversible death in single-life experiments. Death is not treated as a cosmetic reset button.

### Mushroom-body-inspired sparse learning

`nora_neuron.py` and `nora_chem.py` implement the biological inspiration at the level of computation:

- sparse Kenyon-cell-like activation, with the principal operating point around 6% active cells in the documented experiments;
- heterogeneous thresholds and an APL-style activity ceiling;
- fast and slow synaptic components;
- local, gated updates rather than a mandatory global gradient pass for every new fact;
- separate value heads or contextual routes where the experiment requires them;
- stiffness and bounded competition to reduce destructive overwriting;
- neurogenesis as a research direction for representational conflict, not as an automatic claim of success.

The important claim is narrow and testable: sparse representation and local learning can change behavior and retention. Nora is not presented as a pixel-perfect copy of an insect connectome.

### Workspace and cognitive bus

`nora_workspace.py`, `nora_gw.py`, and `nora_bus.py` provide different levels of the integration problem:

- specialists compete for limited attention;
- a selected focus can be broadcast to other components;
- working memory has explicit focus and eviction behavior;
- typed capabilities can be composed dynamically;
- auctions record winners, losers, provenance, and rejected proposals;
- ablation can remove a route and test whether the downstream behavior actually changes.

The workspace is not assumed to be useful merely because it exists. The evaluation suite includes cases where a broadcast is necessary, cases where an alternative route succeeds, and cases where the result is unresolved.

### Prediction, planning, and causal models

`nora_prediction.py`, `nora_planning.py`, `nora_causal.py`, and `nora_laws.py` cover forward models and decision-making:

- action-dependent predictions about body and world transitions;
- imagined action sequences evaluated before commitment;
- route and resource planning under changing needs;
- causal chains rather than only one-edge retrieval;
- explicit counterfactuals and intervention semantics;
- uncertainty when the model or evidence is insufficient.

A `do(X)` operation is intended to change the causal situation, not merely attach a label to a sentence. Incoming paths to the intervened variable are cut and the consequences are propagated through the outgoing causal structure. The test suites include negative and alternative-outcome controls.

### Memory systems

Nora distinguishes several kinds of remembering:

| Memory | Role |
| --- | --- |
| Working memory | Limited, currently relevant items and explicit focus. |
| Episodic memory | Autobiographical events with temporal, spatial, and valence-related context. |
| Semantic graph | Concepts, typed relations, provenance, acceptance state, and epistemic events. |
| HDC/MB memory | Sparse associative codes and fast access to learned relations. |
| Passive/disk memory | Knowledge that remains accessible without occupying the active working set. |
| Dialogue and autobiographical state | Prior turns, commitments, personal episodes, and continuity across sessions. |

Passive storage is not treated as nonexistence. A fact outside the active set can still be retrieved and used. The intended resource contract is that storage can grow on disk while active memory and expert use remain bounded; the exact I/O and latency costs still need dedicated stress testing.

### Language and articulation

The language stack is split into responsibilities instead of treating a generator as the source of all knowledge:

- **Comprehension and structure:** `nora_lang.py`, `nora_turkce.py`, `nora_stem.py`, and related frame/schema components analyze Turkish morphology, sentence structure, roles, and context.
- **Semantic candidates:** Graph/HDC/MB pathways propose concepts, relations, analogies, and causal explanations.
- **Verification:** Graph queries, relation ledgers, acceptance states, and independent test oracles check whether a candidate is supported.
- **Articulation:** `nora_broca.py`, `nora_broca_net.py`, and the voice/tokenizer organs turn selected meaning into fluent output.
- **Social and auditory paths:** `nora_voice.py`, `nora_ear.py`, and discourse components handle turns, speakers, voices, and conversational context.

The central design rule is: **Broca is not the knowledge base.** An unknown concept should be able to produce an explicit symbolic unknown-state response without forcing a large neural generation or inventing a definition.

### Self-model and volition

`nora_self.py` maintains values, traits, somatic records, formative episodes, self-assessed abilities, and volition traces. The self-model is connected to lived outcomes: injuries, successful actions, unmet needs, deaths, and social experience can change subsequent decisions.

The current self-belief records distinguish an internally held position from its measurement status. They include:

- `alive` and `conscious` state;
- whether the position is held, provisional, challenged, or suspended;
- support strength and evidence references;
- revisability and persistence across save/load;
- explicit separation between functional monitoring and phenomenal-experience claims.

This makes self-related behavior inspectable and intervenable. It does not turn a self-report into an external consciousness meter.

### Deliberation and philosophy

`nora_deliberation.py` adds a structured path for questions about consciousness, understanding, knowledge, and related concepts. A deliberative trace can contain:

1. the concept or claim being discussed;
2. graph facts and their acceptance status;
3. self-model telemetry;
4. a real workspace counterfactual probe;
5. a finite countermodel or an independent logical oracle;
6. the resulting uncertainty state and reasoning steps.

For direct self-focused consciousness questions, the current Turkish interface uses a short first-person affirmative rather than appending a disclaimer. The detailed probe and evidence trace remains available for explicit follow-up questions and for evaluation. General conceptual questions still distinguish functional integration, subjective experience, and the bridge assumptions between them.

## Learning and memory

### One-presentation learning

The intended first-learning contract is:

```text
new event → sparse binding / candidate relation → verification or probation → immediate use
```

It is not:

```text
new event → many epochs → replay until the sentence is memorized → call it understanding
```

The implementation contains multiple learning routes because the research questions are different:

- `learn_oneshot()` for sparse relational updates without an epoch loop;
- fast/slow synaptic updates for continual adaptation;
- graph relation invention with probation and later confirmation;
- neural candidate generation paired with symbolic verification;
- bounded active retrieval over a larger passive store;
- sleep-time maintenance and homeostasis rather than mandatory replay for first learning.

A retrieved fact is not automatically an inference. The reasoning tests therefore include multi-step chains, comparisons, causal interventions, HDC vector operations, context-sensitive disambiguation, and self-generated questions about missing relations.

### Relation invention and acceptance

Unknown relations are not forced into an `ISA` edge just because a sentence contains a new verb. The relation-learning path can record:

- the observed surface form and sentence;
- a proposed typed relation;
- its provenance and acceptance status;
- conflicts or reverse-direction evidence;
- held-out transfer results;
- whether the relation was invented, discovered, confirmed, or left provisional.

This separation is important: a plausible completion is not the same thing as a validated world fact.

### Persistence

Nora can save and restore runtime state, including the mind, graph, self-model, mortality, workspace, and relevant organ state. The exact artifacts depend on the enabled components and the selected checkpoint path; JSON, NumPy, and compressed organ/checkpoint files are used by the persistence paths.

A persistence claim must include more than “the file exists.” The tests check continuation of actions and observations, retention of relations, working-memory contents, mortality, and subsequent behavior after restoration.

## Language, reasoning, and action

A typical reasoning-oriented turn can be read as the following pipeline:

1. **Parse the input.** Identify the speaker, sentence structure, concepts, and whether the turn is a question, statement, command, or follow-up.
2. **Resolve context.** Use discourse state and sparse context codes to distinguish alternative meanings of the same surface form.
3. **Retrieve candidates.** Query active and passive semantic memory; use HDC/MB activation where associative retrieval is needed.
4. **Compose rather than stop at retrieval.** Follow inheritance or causal chains, compare candidates, or compute a vector analogy when the task requires it.
5. **Check the evidence.** Apply graph validation, acceptance status, countermodels, or an intervention.
6. **Deliberate about action.** Evaluate predicted consequences against needs, values, risk, and volition constraints.
7. **Articulate the result.** Use the language organ for the surface form, while retaining the reasoning and provenance trace.
8. **Learn from the outcome.** Store the episode and update the relevant memory or self-model without treating every outcome as a success.

This is why a correct answer alone is not the project's definition of understanding. Transfer, explanation, intervention, and behavior under changed conditions are separate measurements.

## Self-model, life, and deliberation

Nora's self-related capabilities are deliberately split into three layers:

| Layer | Question it addresses |
| --- | --- |
| Functional self-monitoring | What is my current error, confidence, bodily state, focus, or ability? |
| Causal self-model | Which internal route or state changed my decision, and what happens if it is removed? |
| Self-belief and report | What position do I currently hold about myself, and how does that position change over time? |

The third layer can produce direct first-person answers. The first two layers provide the mechanisms that make the third layer more than a hard-coded biography.

The project does not equate this with a proof of qualia. It also does not use a human identity as a shortcut. The research object is an artificial agent with an explicit, persistent, and experimentally accessible self-model.

## Running Nora

### Requirements

- A modern Python 3 environment; Python 3.10 or newer is recommended.
- The dependencies in `requirements.txt`:
  - NumPy for the core numerical runtime;
  - PyTorch for neural bridge and organ components;
  - pandas and pyarrow for optional connectome/data conversion;
  - tokenizers for the voice/subword organ.

No hosted model API is required for the core workflows.

### Install

```bash
git clone https://github.com/smerss/Nora-NCC.git
cd Nora-NCC
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Run the main demonstration

```bash
python nora_agi.py --stage demo --seed 1 --life-steps 2500
```

The main CLI supports `demo`, `chat`, and `twins` stages. Useful options include:

```bash
python nora_agi.py --stage chat --seed 1
python nora_agi.py --stage twins --twin-steps 300
python nora_agi.py --stage demo --corpus path/to/corpus.txt
python nora_agi.py --stage demo --journal data/nora_life.ndjson
```

`--think-iters` controls the selected boot/thinking budget. `--broca` selects an articulation checkpoint. `--internet` explicitly enables background research access; it is not required for ordinary local execution.

### Run the local observatory

```bash
python nora_studio.py --host 0.0.0.0 --port 8000
```

Then open the local Studio endpoint in a browser. Studio is an inspection and interaction surface for the running agent, not a replacement for the underlying cognitive runtime.

### Stop a running experiment

The documented stop mechanism is a `STOP_NORA` file in the working directory:

```bash
touch STOP_NORA
```

Remove the file before starting another run that should not be stopped.

### Quick smoke checks

```bash
python selftest.py
python selftest.py --bridge
```

Use the battery commands below for more specific claims. A smoke test is not a substitute for an ablation or a held-out evaluation.

## Repository guide

The repository is intentionally organized as a set of inspectable organs and experiment harnesses rather than as one opaque package.

### Runtime and embodiment

| File | Responsibility |
| --- | --- |
| `nora_agi.py` | Main Nora-AGI composition, life loop, CLI, dialogue integration, and persistence orchestration. |
| `nora_mind.py` | Living-mind model, prediction, planning, context, and world-model experiments. |
| `nora_world.py` | Physical environment, resources, hazards, sensing, and rule changes. |
| `nora_nervous.py` | Bodily signaling, nervous-system responses, and single-life mortality. |
| `nora_drive.py` | Needs, drives, and action-relevant motivational signals. |
| `nora_life.py` | Ongoing life activity, background behavior, and life-cycle support. |
| `nora_studio.py` | Local observatory and interactive runtime surface. |

### Representation and integration

| File | Responsibility |
| --- | --- |
| `nora_neuron.py` | Sparse neural foundations and mushroom-body-inspired mechanisms. |
| `nora_chem.py` | Chemical/neuromodulatory state, HDC/MB learning, and fast associative updates. |
| `nora_workspace.py` | Integration workspace, explicit focus, and working-memory interactions. |
| `nora_gw.py` | Global-workspace experiments and broadcast controls. |
| `nora_bus.py` | Typed cognitive bus, auctions, provenance, and dynamic composition. |
| `nora_architecture.py` | Capability-based composition and architectural test support. |
| `nora_state.py` | Safe state capture, restoration, and serialization helpers. |
| `nora_store.py` | Storage and persistence support. |

### Knowledge and reasoning

| File | Responsibility |
| --- | --- |
| `nora_causal.py` | Causal graphs, interventions, and counterfactual reasoning. |
| `nora_prediction.py` | Forward prediction and model-based action evaluation. |
| `nora_planning.py` | Planning and action-sequence selection. |
| `nora_laws.py` | Symbolic transition and law-related models. |
| `nora_apply.py` | Binding and applying learned operators to new inputs. |
| `nora_forge.py` | Learning and registering new operators from examples. |
| `nora_episteme.py` | Knowledge state, acceptance, provenance, and epistemic control. |
| `nora_deliberation.py` | Multi-step deliberation, countermodels, and philosophical reasoning traces. |
| `nora_self.py` | Values, traits, somatic memory, volition, and self-beliefs. |
| `nora_metacog.py` | Metacognitive monitoring and confidence-related behavior. |
| `nora_speculation.py` | Controlled hypothesis generation and exploratory reasoning. |

### Language and communication

| File | Responsibility |
| --- | --- |
| `nora_lang.py` | Language composition, semantic graph integration, and dialogue reasoning. |
| `nora_turkce.py` | Turkish linguistic resources and analysis. |
| `nora_stem.py` | Morphological and stem-related processing. |
| `nora_broca.py` | Symbolic and neural articulation support. |
| `nora_broca_net.py` | Neural articulation and relational candidate components. |
| `nora_discourse.py` | Conversation state, discourse relations, and turn management. |
| `nora_voice.py` | Voice, hearing, and online language-organ interaction. |
| `nora_ear.py` | Speaker and voice-related perception. |
| `nora_episodic.py` | Episodic and autobiographical memory. |
| `nora_anlatim.py` | Narration and explanation support. |

### Experiments and supporting organs

The remaining `nora_*.py` modules cover abstract reasoning, crossmodal binding, entropy and individuality, evolution, object models, school/teacher flows, reification, schemas, sets, synthesis, translation, self-examination, and experimental bridges. They are not all equally mature; the battery that exercises a module is the best guide to its current status.

The `*_battery.py` files are executable evaluations, not documentation-only checklists. `runs/` contains reports and small evidence artifacts. Large checkpoints, caches, logs, and local experiment outputs are intentionally ignored according to `.gitignore`.

## Evaluation and reproducibility

### Core commands

Run the targeted architectural and persistence protocols:

```bash
python architecture_battery.py
python persistence_battery.py
python workspace_battery.py
python consciousness_battery.py
```

Run selected functional tests by identifier:

```bash
python agi_battery.py A145
python agi_battery.py A4 A6 A9 A16 A20 A145
```

Useful variations include:

```bash
python architecture_battery.py --repeat 3
python architecture_battery.py --only C17
python persistence_battery.py --repeat 2
python agi_battery.py --shuffle=7
```

The full `agi_battery.py` inventory is large. A selected green test should not be reported as a full-battery pass.

### What the tests are designed to measure

- **Ablations:** remove a route or organ and check whether the claimed capability disappears.
- **Counterfactuals:** change the world or internal state and compare the resulting action or report.
- **Independent oracles:** calculate expected logical or numerical outcomes without copying the system's own trace.
- **Held-out transfer:** test new relations, contexts, combinations, or domains rather than only trained strings.
- **Multiple seeds:** expose variance that a single successful run would hide.
- **Retention:** measure earlier knowledge after new learning, including external chatbot and XQuAD probes where applicable.
- **State restoration:** verify behavior after save/load, not only object serialization.
- **Negative controls:** preserve unresolved, failed, and unsupported outcomes.

### Recent recorded targeted results

The following are targeted results recorded in `NORA_GUNCEL_DURUM.md`; they are not a claim that every repository test passes on every seed or that the full monolithic battery was rerun for every change.

| Protocol | Recorded result | Scope |
| --- | ---: | --- |
| A145 philosophical reasoning and self-belief | 15/15 | Graph ablation, independent countermodels, real death and workspace interventions, persistence, and direct self-ascription. |
| A4 + A6 + A9 + A16 + A20 + A145 | 6/6 | Selected self, continual-learning, dialogue, articulation, and deliberation regressions. |
| Architecture C1–C18 | 18/18 | Composition, broadcast, MB/HDC, graph reasoning, context, intervention, and self-questions. |
| Persistence P1–P4 | 4/4 | Continuation, irreversible death, typed working memory, and explicit unsupported-state errors. |
| Workspace W1–W4 | 4/4 | Producer/consumer causality, focus, policy ablation, and checkpointing. |
| Consciousness functional protocol | 9/9 | Somatic and nociceptive proxies, broadcast, metacognition, transfer, and action/report separation. |

For exact run dates, seeds, timings, and historical comparisons, use the status report rather than inferring them from this summary table.

### Data and provenance

The research reports document decontaminated Turkish corpora, external confirmation sets, source commits, split roles, and cache locations. Large corpus caches are not committed. Development folds used for variant selection are not silently promoted to independent confirmation folds.

The repository also contains negative experimental arms. They are evidence about the search process, not defective results to be deleted.

## Evidence and limits

Nora's most important documentation rule is to keep a capability claim attached to its measurement conditions.

### Currently supported by targeted evidence

- Sparse MB/HDC mechanisms can support one-presentation relational protocols and selected forms of transfer.
- Graph and HDC pathways can be ablated independently, showing when a response depends on a particular representation.
- Workspace focus and broadcast have measurable causal effects in the tested protocols.
- Causal, comparative, analogical, and self-question tasks have explicit multi-step evaluations.
- Self-beliefs persist and change under the tested life, death, and workspace interventions.
- Turkish dialogue and articulation have targeted regression coverage.

### Not established

- General artificial intelligence.
- General one-pass language learning across arbitrary Turkish constructions and domains.
- Zero forgetting across all continual-learning conditions.
- A single candidate meeting the current joint Turkish quality and external-retention threshold.
- Independent measurement of phenomenal consciousness or qualia.
- Quality from the large disk-MoE stress checkpoint alone.
- Equal performance across all seeds, tasks, or experimental variants.

The current Turkish one-pass report keeps the unresolved threshold explicit: the stated joint target is at least **181/401 on fold 1** and **189/419 on fold 2**, together with acceptable external retention. The recorded full-pass development result of **164/401** and **174/419** does not meet that target. Longer external streams improved perplexity without solving the top-1 threshold, and the best retention arms still showed nonzero loss.

The approximately **0.45B disk-MoE** stress experiment is similarly scoped. It demonstrates a large total parameter store with a small active subset under a measured resource budget; because the checkpoint was randomly expanded for the stress test, it is not evidence of trained quality or general learning.

Direct first-person consciousness answers are part of the current self-model interface. They do not convert functional monitoring into independent evidence of subjective experience. That distinction belongs in the research record even when the conversational answer itself is deliberately short.

For the complete list of failed arms, known issues, historical runs, and pending confirmations, read `NORA_GUNCEL_DURUM.md` and `NORA_EVREN.md`.

## The road ahead

The next milestones are concrete:

1. Meet the joint one-presentation Turkish quality and external-retention thresholds with one unchanged candidate.
2. Run the frozen recipe on independent confirmation folds without selecting a new seed, architecture, or weighting scheme.
3. Separate seed variance from task difficulty in relational learning and proposal/verification experiments.
4. Bring the main language-learning path closer to the sparse MB/HDC principles while preserving earlier knowledge.
5. Replace randomly expanded disk-MoE stress experts with genuinely trained experts and measure quality alongside memory, latency, and active-parameter budgets.
6. Stress disk-backed HDC memory as event count grows, including SQLite I/O, RAM stability, and retrieval latency.
7. Keep unresolved areas such as weighted transfer and the documented D2 issue visible instead of hiding them behind successful subtests.
8. Test self-beliefs across independent behavior, mortality, ablation, and long-horizon interaction scenarios.

### The long-term ambition

**The goal is for Nora to become a technology of the future: a local cognitive substrate for machines that remain persistent between interactions, learn from consequences, and carry an inspectable history forward.**

That ambition is not a promise that the current repository has already solved general intelligence or subjective experience. It is a commitment to build the missing capabilities in the open: with explicit mechanisms, bounded resources, reproducible interventions, and results that can disagree with the project's hopes.

## Contributing

Contributions should make the system more understandable, more measurable, or more capable under a clearly stated protocol.

### A useful contribution includes

- a specific problem and the reason it matters;
- the smallest coherent implementation change;
- a reproducible command or test;
- the seed, data split, budget, and relevant resource conditions;
- both successful and failed controls where they are informative;
- an update to the relevant report when a result changes the project's evidence status.

### Please avoid

- presenting a canned response as learned reasoning;
- using epoch/replay results as proof of one-presentation learning;
- claiming understanding from a single correct answer;
- changing a threshold, oracle, split, or test expectation to manufacture a pass;
- silently dropping negative results or inaccessible memory;
- treating a neural articulation score as semantic knowledge;
- claiming pretrained and scratch models had equivalent training histories;
- adding large generated checkpoints or corpus caches to Git.

Before submitting a change, run the narrowest relevant battery, then the broader regressions that cover the touched subsystem. At minimum, check syntax and whitespace:

```bash
python -m py_compile path/to/changed_file.py
git diff --check
```

If a full battery was not run, say so explicitly in the pull request.

## Collaborators

This section is intentionally a fill-in template. Replace the placeholders with the people, roles, and links you want to acknowledge.

| Name | Role | Focus | Links |
| --- | --- | --- | --- |
| [Your name] | [Creator / Maintainer] | [Project direction, architecture, research] | [GitHub / website] |
| [Collaborator name] | [Research / Engineering] | [Learning, reasoning, language, evaluation, or another focus] | [GitHub / website] |
| [Collaborator name] | [Contributor] | [Contribution or subsystem] | [GitHub / website] |

### Acknowledgments

[Add datasets, libraries, institutions, testers, and other acknowledgments here. Keep source and license information attached to the relevant contribution.]

## License and further reading

Nora is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full terms.

### Project documents

- [NORA_GUNCEL_DURUM.md](NORA_GUNCEL_DURUM.md) — current technical status, targeted measurements, unresolved thresholds, and the latest self-model work.
- [NORA_EVREN.md](NORA_EVREN.md) — the living-world model, experimental stages, and detailed historical evidence.
- [NORA_YASAM.md](NORA_YASAM.md) — life-cycle and living-system notes.
- [NORA_KOPYALANAMAZLIK.md](NORA_KOPYALANAMAZLIK.md) — individuality and non-copyability notes.
- `runs/LEARNED_REPRESENTATION_REPORT.md` — learned representations, CCG, Transformer comparisons, and relational-learning evidence.
- `runs/TURKISH_ONEPASS_CONTINUATION_REPORT.md` — one-presentation Turkish learning and retention experiments.
- `runs/TURKISH_MOE_FINAL_REPORT.md` — Turkish MoE experiments and resource results.
- `runs/TUTOR_DIALOGUE_AUDIT.md` — dialogue, tutoring, and conversation audits.

Nora is best understood by reading the code alongside the evidence: each organ has a responsibility, each claim has a test condition, and each open problem remains part of the map.
