def get_metrics_help_modal():
    modal_html = """
<div id="metricsHelpModal" class="modal">
  <div class="modal-content">
    <span class="close">×</span>
    <h2>Model Evaluation Metrics — Help Guide</h2>
    <div class="metrics-guide">
      <h3>1) General Metrics</h3>
      <p><strong>Loss:</strong> Measures the difference between predicted and actual values. Lower is better. Often used for optimization during training.</p>
      <p><strong>Accuracy:</strong> Proportion of correct predictions among all predictions. Simple but can be misleading for imbalanced datasets.</p>
      <p><strong>Micro Accuracy:</strong> Calculates accuracy by summing up all individual true positives and true negatives across all classes, making it suitable for multiclass or multilabel problems.</p>
      <p><strong>Token Accuracy:</strong> Measures how often the predicted tokens (e.g., in sequences) match the true tokens. Useful in sequence prediction tasks like NLP.</p>
      <h3>2) Precision, Recall & Specificity</h3>
      <p><strong>Precision:</strong> Out of all positive predictions, how many were correct. Precision = TP / (TP + FP). Helps when false positives are costly.</p>
      <p><strong>Recall (Sensitivity):</strong> Out of all actual positives, how many were predicted correctly. Recall = TP / (TP + FN). Important when missing positives is risky.</p>
      <p><strong>Specificity:</strong> True negative rate. Measures how well the model identifies negatives. Specificity = TN / (TN + FP). Useful in medical testing to avoid false alarms.</p>
      <h3>3) Macro, Micro, and Weighted Averages</h3>
      <p><strong>Macro Precision / Recall / F1:</strong> Averages the metric across all classes, treating each class equally, regardless of class frequency. Best when class sizes are balanced.</p>
      <p><strong>Micro Precision / Recall / F1:</strong> Aggregates TP, FP, FN across all classes before computing the metric. Gives a global view and is ideal for class-imbalanced problems.</p>
      <p><strong>Weighted Precision / Recall / F1:</strong> Averages each metric across classes, weighted by the number of true instances per class. Balances importance of classes based on frequency.</p>
      <h3>4) Average Precision (PR-AUC Variants)</h3>
      <p><strong>Average Precision Macro:</strong> Precision-Recall AUC averaged across all classes equally. Useful for balanced multi-class problems.</p>
      <p><strong>Average Precision Micro:</strong> Global Precision-Recall AUC using all instances. Best for imbalanced data or multi-label classification.</p>
      <p><strong>Average Precision Samples:</strong> Precision-Recall AUC averaged across individual samples (not classes). Ideal for multi-label problems where each sample can belong to multiple classes.</p>
      <h3>5) ROC-AUC Variants</h3>
      <p><strong>ROC-AUC:</strong> Measures model's ability to distinguish between classes. AUC = 1 is perfect; 0.5 is random guessing. Use for binary classification.</p>
      <p><strong>Macro ROC-AUC:</strong> Averages the AUC across all classes equally. Suitable when classes are balanced and of equal importance.</p>
      <p><strong>Micro ROC-AUC:</strong> Computes AUC from aggregated predictions across all classes. Useful in multiclass or multilabel settings with imbalance.</p>
      <h3>6) Ranking Metrics</h3>
      <p><strong>Hits at K:</strong> Measures whether the true label is among the top-K predictions. Common in recommendation systems and retrieval tasks.</p>
      <h3>7) Confusion Matrix Stats (Per Class)</h3>
      <p><strong>True Positives / Negatives (TP / TN):</strong> Correct predictions for positives and negatives respectively.</p>
      <p><strong>False Positives / Negatives (FP / FN):</strong> Incorrect predictions — false alarms and missed detections.</p>
      <h3>8) Other Useful Metrics</h3>
      <p><strong>Cohen's Kappa:</strong> Measures agreement between predicted and actual values adjusted for chance. Useful for multiclass classification with imbalanced labels.</p>
      <p><strong>Matthews Correlation Coefficient (MCC):</strong> Balanced measure of prediction quality that takes into account TP, TN, FP, and FN. Particularly effective for imbalanced datasets.</p>
      <h3>9) Metric Recommendations</h3>
      <ul>
        <li>Use <strong>Accuracy + F1</strong> for balanced data.</li>
        <li>Use <strong>Precision, Recall, ROC-AUC</strong> for imbalanced datasets.</li>
        <li>Use <strong>Average Precision Micro</strong> for multilabel or class-imbalanced problems.</li>
        <li>Use <strong>Macro scores</strong> when all classes should be treated equally.</li>
        <li>Use <strong>Weighted scores</strong> when class imbalance should be accounted for without ignoring small classes.</li>
        <li>Use <strong>Confusion Matrix stats</strong> to analyze class-wise performance.</li>
        <li>Use <strong>Hits at K</strong> for recommendation or ranking-based tasks.</li>
      </ul>
    </div>
  </div>
</div>
"""
    modal_css = """
<style>
.modal {
  display: none;
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.4);
}
.modal-content {
  background-color: #fefefe;
  margin: 15% auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
  max-width: 800px;
}
.close {
  color: #aaa;
  float: right;
  font-size: 28px;
  font-weight: bold;
}
.close:hover,
.close:focus {
  color: black;
  text-decoration: none;
  cursor: pointer;
}
.metrics-guide h3 {
  margin-top: 20px;
}
.metrics-guide p {
  margin: 5px 0;
}
.metrics-guide ul {
  margin: 10px 0;
  padding-left: 20px;
}
</style>
"""
    modal_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
  var modal = document.getElementById("metricsHelpModal");
  var openBtn = document.getElementById("openMetricsHelp");
  var span = document.getElementsByClassName("close")[0];
  if (openBtn && modal) {
    openBtn.onclick = function() {
      modal.style.display = "block";
    };
  }
  if (span && modal) {
    span.onclick = function() {
      modal.style.display = "none";
    };
  }
  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
});
</script>
"""
    return modal_css + modal_html + modal_js

