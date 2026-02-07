# # predict.py
# # Works with PIL images from app.py, returns top-k predictions,
# # and provides abstain/unknown detection (float OR topk-list compatible).

# import numpy as np
# from PIL import Image


# def _to_model_input(img, size=(224, 224)):
#     """
#     Accepts a PIL.Image (what Streamlit gives you) and returns a batch tensor:
#       shape: (1, 224, 224, 3)
#       dtype: float32
#       range: [0, 1]
#     """
#     if not isinstance(img, Image.Image):
#         img = Image.open(img)

#     img = img.convert("RGB").resize(size)
#     x = np.asarray(img, dtype=np.float32) / 255.0
#     x = np.expand_dims(x, axis=0)
#     return x


# def _softmax_if_needed(preds_1d):
#     """
#     If model outputs probabilities, return as-is.
#     If model outputs logits, convert to probabilities with softmax.
#     """
#     p = np.asarray(preds_1d, dtype=np.float32)

#     s = float(p.sum())
#     looks_like_probs = (0.99 <= s <= 1.01) and np.all(p >= 0.0) and np.all(p <= 1.0)
#     if looks_like_probs:
#         return p

#     # logits -> softmax
#     p = p - np.max(p)
#     exp = np.exp(p)
#     return exp / np.sum(exp)


# def predict_topk(model, img, class_names, k=3, input_size=(224, 224)):
#     """
#     Returns (exactly 4 values, so your app.py can do: top1, conf, top3, _):
#       top1_label (str)
#       top1_conf  (float)
#       topk_list  (list of (label, conf))
#       probs      (np.ndarray full vector)
#     """
#     x = _to_model_input(img, size=input_size)

#     raw = model.predict(x, verbose=0)[0]
#     probs = _softmax_if_needed(raw)

#     k = int(k)
#     top_idx = np.argsort(probs)[::-1][:k]
#     topk = [(class_names[int(i)], float(probs[int(i)])) for i in top_idx]

#     top1_label, top1_conf = topk[0]
#     return top1_label, top1_conf, topk, probs

# def should_abstain(topk_list, threshold=0.30, margin=0.05):
#     if not topk_list or len(topk_list) < 2:
#         return True
#     top1 = float(topk_list[0][1])
#     top2 = float(topk_list[1][1])

#     # Only abstain when BOTH are true:
#     return (top1 < threshold) and ((top1 - top2) < margin)
