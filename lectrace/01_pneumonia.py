"""Five selected execution states using the supplied trained model.
Inspired by the official LecTrace demo https://github.com/praisegee/lectrace-demo.
AI-assisted authoring; original project weights; no training or test evaluation.
"""
from lectrace import text, note, table, image
from _helpers import run_demo, entropy

def main():
    text('# Confidence and disagreement in pneumonia classification')
    text('We study a research benchmark, not clinical diagnosis. Can an uncertainty score identify predictions that should leave automatic processing?')
    text('A model that predicts pneumonia for almost every image can have high sensitivity while creating many false alarms. We compare classification performance with the ability to select errors for review.')
    note('0:00–1:15 | Explain false alarms: high sensitivity alone does not mean good overall performance.')
    text('Study design: PneumoniaMNIST, 28 × 28 grayscale images; official test set: 624 images (234 normal, 390 pneumonia). Baseline, blur augmentation and Dropout variants; 4,866 parameters; three training seeds; ten epochs per run. The protocol was frozen after exploratory validation.')
    note('1:15–2:30 | Introduce the data, three model variants and three seeds. Settings were frozen after validation exploration.')
    text('Research questions: RQ1 — How does Gaussian blur change classification ranking and class-specific errors? RQ2 — With identical MC predictions, does low-confidence or high-MI selection retain fewer errors at the same review budget?')
    text('Course connection: MD6329 Lecture 2, slides 19 and 57 explain CNN feature extraction; slides 84 and 92 connect confusion-matrix metrics with ROC and threshold trade-offs. These are verified slide references, not video timestamps.')
    state = run_demo()  # @stepover @hide

    text('## State 1 — The input changes')
    text('What happens: the same validation images are blurred. What changes: fine pixel detail is reduced while labels stay fixed. Why it matters: this isolates a controlled input degradation.')
    input_shape = state['input_shape']  # @inspect input_shape
    image('assets/inputs.png', width=480)
    text('These are fixed validation examples. Gaussian blur changes the pixels while the dataset labels stay fixed. We do not use images to diagnose a patient.')
    note('2:30–3:20 | Point to lost detail. Pixels change; stored labels stay fixed.')

    text('## State 2 — The network computes features')
    text('What happens: learned convolutions, ReLU and pooling compute image features. What changes: pixels become channel activations and then a 32-number representation. Why it matters: these features determine the classifier output.')
    convolution_shape = state['conv_shape']  # @inspect convolution_shape
    pooled_shape = state['pooled_shape']  # @inspect pooled_shape
    image('assets/features.png', width=480)
    text('The first convolution produces 16 feature maps. Two convolution blocks and average pooling reduce each image to 32 numbers. Feature maps are not verified lesion explanations.')
    note('3:20–4:20 | Follow the tensor shapes from pixels to 32 features. Activations are not verified lesion maps.')

    text('## State 3 — Dropout changes repeated predictions')
    text('What happens: only the Dropout layer is enabled for repeated inference. What changes: the feature mask and predicted probabilities vary across passes; weights stay fixed. Why it matters: these variations provide the inputs for MI.')
    first_logits = state['first_logits']  # @inspect first_logits
    first_five_probabilities = state['first_5_draws']  # @inspect first_five_probabilities
    text('The logits above come from deterministic inference. We then enable only Dropout and compute 30 softmax probability vectors. We average probabilities, not logits. These small-batch draws are illustrative and differ from full-evaluation draws.')
    note('4:20–5:30 | Explain 30 stochastic predictions with unchanged weights; average probabilities, not logits.')

    text('## State 4 — Confidence and MI measure different things')
    text('What happens: 30 probability vectors are summarized. What changes: predictions become confidence and MI scores. Why it matters: proximity to the decision boundary and disagreement need not rank errors in the same order.')
    mean_probability = sum(state['first_draws']) / 30  # @inspect mean_probability
    predictive_entropy = float(entropy(mean_probability))  # @stepover @inspect predictive_entropy
    expected_entropy = float(entropy(state['first_draws']).mean())  # @stepover @inspect expected_entropy
    mutual_information = predictive_entropy - expected_entropy  # @inspect mutual_information
    table(state['records'], caption='Actual computations on eight fixed validation examples; not aggregate test performance')
    text('Confidence is max(q, 1-q). MI is H(q) minus the average H(p_t). Low confidence means the mean prediction is near the decision boundary; MI reflects disagreement among stochastic predictions. Neither is a clinical risk probability.')
    note('5:30–6:50 | Contrast low confidence with disagreement. Neither score is a clinical risk probability.')

    text('## State 5 — A fixed budget produces a selection')
    text('What happens: each score ranks the same predictions for review. What changes: which images leave automatic processing. Why it matters: compare retained errors at an identical workload, without assuming human review is correct.')
    confidence_review = state['B_first_two']  # @inspect confidence_review
    mi_review = state['C_first_two']  # @inspect mi_review
    text('For this eight-example illustration only, two images are reviewed, a 25% budget. The formal test experiment instead reviews 124 of 624 images and retains 500. B and C share the same predicted labels and differ only in selection order.')
    table([{'sigma':0,'B risk':.1580,'C risk':.1853,'C minus B pp':2.7333},{'sigma':.5,'B risk':.2240,'C risk':.2367,'C minus B pp':1.2667},{'sigma':1,'B risk':.2607,'C risk':.2820,'C minus B pp':2.1333},{'sigma':1.5,'B risk':.2760,'C risk':.3160,'C minus B pp':4.0}],caption='Previously saved final-test means across three seeds; not recomputed by this demo')
    note('6:50–8:10 | Compare the two review lists. Separate the eight-image illustration from the 624-image test.')
    text('## Main findings')
    text('Evidence scope: the walkthrough above uses eight fixed validation examples and a 2/8 illustrative review budget. The following figures use the saved final test results: 624 test images, three training seeds; no test inference is performed here.')
    image('assets/test_classification.png', width=850)
    text('Figure 1 — Saved test means ± sample SD across three seeds. Left: AUROC. Right: specificity at threshold 0.5. SD is not a population confidence interval. Compare ranking quality with the ability to recognize normal images.')
    text('Under strong blur (sigma 1.5), baseline AUROC remains 0.8504, but specificity falls to zero: every test image is classified as pneumonia. Blur augmentation does not prevent near-universal positive predictions; MC averaging does not consistently improve classification.')
    image('assets/test_review.png', width=850)
    text('Figure 2 — Saved test retained risk after reviewing 124/624 images, retaining 500 (80.13% coverage). Lower is better. Error bars: sample SD across three seeds, not confidence intervals. The random reference matches the same predictions.')
    text('For the same MC predictions, low-confidence selection retains fewer errors than high-MI selection in all three seeds and all four conditions. At sigma 1.5, mean retained risk is 27.6% versus 31.6%: a paired difference of 4.00 percentage points. Both means are below the matched theoretical random reference (37.45%), but substantial errors remain.')
    note('8:10–9:15 | Point to the saved results: at sigma 1.5, B risk is 27.6% and C risk is 31.6%.')
    text('## Interpretation and limitations')
    text('This result concerns one small-image dataset, one corruption type, a compact network and a ten-epoch training budget. Standard deviations across three seeds on the same test set are not population confidence intervals. We did not measure human review accuracy or establish clinical safety; MI is not universally inferior.')
    note('9:15–10:00 | State the narrow conclusion and limitations. Do not claim clinical safety or universal superiority.')
    text('Final message: benchmark complex uncertainty scores against simple confidence using the decision objective you actually care about. This is a controlled study, not evidence that MI is universally inferior or that a review system is clinically safe.')

if __name__=='__main__':
    main()
