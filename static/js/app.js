import { requestPrediction } from "./api.js";
import { FIELDS, MALIGNANT_DEMO } from "./config.js";
import { readValidFeatures } from "./validation.js";
import { renderError, renderLoading, renderPrediction, setSubmitting } from "./ui.js";

const form = document.querySelector("#prediction-form");

async function handleSubmit(event) {
  event?.preventDefault();

  try {
    const features = readValidFeatures(form, FIELDS);
    setSubmitting(true);
    renderLoading();
    renderPrediction(await requestPrediction(features));
  } catch (error) {
    renderError(error.message || "Ocorreu um erro inesperado.");
  } finally {
    setSubmitting(false);
  }
}

function fillForm(values) {
  for (const [name, value] of Object.entries(values)) {
    form.elements.namedItem(name).value = value;
  }
}

function runDocumentedDemo() {
  const demo = new URLSearchParams(window.location.search).get("demo");
  if (demo === "success") {
    fillForm(MALIGNANT_DEMO);
    window.setTimeout(handleSubmit, 350);
  }
  if (demo === "error") {
    fillForm({ mean_radius: 45 });
    window.setTimeout(handleSubmit, 350);
  }
}

form.addEventListener("submit", handleSubmit);
runDocumentedDemo();

