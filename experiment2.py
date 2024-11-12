def calculate_metrics(tp, fp, fn):
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"precision, reccall,f1 : {precision},{recall}, {f1}")
    return precision, recall, f1


calculate_metrics(0.3797198190689,0.04610708626508713,0.03285147)
