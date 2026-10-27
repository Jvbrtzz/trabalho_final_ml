export class InputValidationError extends Error {}

export function readValidFeatures(form, fields) {
  const features = {};

  for (const field of fields) {
    const input = form.elements.namedItem(field.name);
    if (!(input instanceof HTMLInputElement)) {
      throw new InputValidationError(`Campo “${field.label}” não encontrado.`);
    }
    if (input.value.trim() === "" || !input.checkValidity()) {
      throw new InputValidationError(
        `Revise “${field.label}”. Use um valor entre ${input.min} e ${input.max}.`
      );
    }
    features[field.name] = Number(input.value);
  }

  return features;
}

