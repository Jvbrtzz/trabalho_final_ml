const PREDICTION_ENDPOINT = "/api/v1/predictions";

export class ApiError extends Error {}

export async function requestPrediction(features) {
  let response;
  try {
    response = await fetch(PREDICTION_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features),
    });
  } catch {
    throw new ApiError("Não foi possível conectar à API. Verifique se o servidor está ativo.");
  }

  if (!response.ok) {
    throw new ApiError(
      response.status === 422
        ? "A API recusou os valores informados. Revise os campos."
        : "A API não conseguiu processar a solicitação."
    );
  }

  return response.json();
}

