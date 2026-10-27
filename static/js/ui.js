const result = document.querySelector("#result");
const submitButton = document.querySelector("#submit-button");

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderMessage(type, iconText, title, message) {
  result.className = `result${type ? ` ${type}` : ""}`;
  result.replaceChildren(
    element("div", iconText, "icon"),
    element("h3", title),
    element("p", message)
  );
}

export function renderLoading() {
  renderMessage("", "…", "Analisando", "Aplicando o pipeline treinado.");
}

export function renderError(message) {
  renderMessage("danger", "!", "Entrada inválida", message);
}

export function renderPrediction(prediction) {
  const percentage = (prediction.probabilidade_maligno * 100).toFixed(1);
  const bar = element("div", undefined, "prob");
  const fill = element("span");
  fill.style.width = `${percentage}%`;
  bar.append(fill);

  const probability = element("p");
  const strong = element("b", `${percentage}%`);
  probability.append(strong, " de probabilidade da classe maligna");

  result.className = "result success";
  result.replaceChildren(
    element("div", "✓", "icon"),
    element("p", "Classe prevista"),
    element("h3", prediction.classe),
    bar,
    probability
  );
}

export function setSubmitting(isSubmitting) {
  submitButton.disabled = isSubmitting;
  submitButton.textContent = isSubmitting ? "Analisando…" : "Analisar amostra →";
}

