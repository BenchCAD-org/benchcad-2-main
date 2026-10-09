# T6 scoring: from correspondence search to reported positions

T6 asks a model to recover a board's schematic from photographs of the board. To
score a submission, its components and nets have to be lined up with the reference
schematic first. On a board that prints reference designators the names can be
read, but some boards print almost none, and there a submission's names are the
model's own invention and say nothing about which reference part they stand for.

**Until now** the alignment was found by search. The scorer looked for the
assignment of submitted components to reference components that made the two
schematics agree best. That search is combinatorial. On most boards it finished,
but it could take up to an hour per case. On the boards with the fewest readable
labels it sometimes could not finish within its budget, and then no score could
be certified.

**Now** the model also reports where each component sits on the board and where
each of its terminals is. The alignment then follows directly from position: a
submitted component is matched to the reference component at the same place. No
search is needed, so scoring always completes, takes milliseconds, and gives the
same answer on any machine. The circuit is judged on the same five things:
components, terminals, nets, shorts and opens. One judgement is stricter: which
pin is which now follows from the reported positions, where the search was free
to choose the arrangement that scored best.

## How the change was checked

We ran the same model on the same boards under both schemes and compared the
results. The scores are close: they differ by about 0.06 on average, and some
boards scored higher under the new scheme. Each board was run once per scheme
and the model is stochastic, so no finer figure is meaningful. The difference
that matters is coverage. The new scheme produced a certified score on every
board in the comparison, including the ones the search could not finish.

Reporting positions does not crowd out the actual work. A model that traces
copper has already located the pads, so writing their coordinates down costs
little, and none of the runs ran short of its call budget.

## How the comparison was run

The comparison was run with a Claude model through **Claude Code**, on a
subscription, inside a sandboxed environment. Its only network access was the
model API, and it held nothing but the task's own inputs: no reference files and
no grading code. Both schemes saw the same board photographs.
