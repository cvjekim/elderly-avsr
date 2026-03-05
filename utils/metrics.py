from typing import Union

import evaluate


_wer_metric = evaluate.load("wer")


def compute_wer(
    predictions: Union[list[str], "EvalPrediction"] = None,
    references: list[str] = None,
    **kwargs,
) -> dict[str, float]:
    # - Handle both standalone call and HF Trainer EvalPrediction format
    if hasattr(predictions, "predictions"):
        eval_pred = predictions
        preds = eval_pred.predictions
        refs = eval_pred.label_ids
        # - Decode token ids if needed (model-specific decoding should happen in collator/model)
        if not isinstance(preds[0], str):
            raise ValueError("Predictions must be decoded to strings before WER computation")
        predictions = preds
        references = refs

    wer_score = _wer_metric.compute(predictions=predictions, references=references)
    return {"wer": wer_score}
