export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly detail: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  return readResponse<T>(response);
}

export async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      Accept: "application/json",
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  return readResponse<T>(response);
}

export async function putJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "PUT",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readResponse<T>(response);
}

export async function patchJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "PATCH",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readResponse<T>(response);
}

export async function deleteJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    method: "DELETE",
    headers: { Accept: "application/json" },
  });
  return readResponse<T>(response);
}

async function readResponse<T>(response: Response): Promise<T> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const rawDetail =
      typeof payload === "object" && payload !== null && "detail" in payload
        ? (payload as { detail: unknown }).detail
        : null;
    const detail =
      typeof rawDetail === "string"
        ? rawDetail
        : Array.isArray(rawDetail)
          ? formatValidationErrors(rawDetail, response.status)
        : rawDetail && typeof rawDetail === "object" && "message" in rawDetail
          ? String((rawDetail as { message: unknown }).message)
          : `Request failed with HTTP ${response.status}.`;
    throw new ApiError(detail, response.status, rawDetail);
  }
  return payload as T;
}

const FIELD_LABELS: Record<string, string> = {
  demos: "Demos",
  display_name: "Route label",
  name: "Name",
  protocol_contract: "Protocol contract",
  public_name: "API Model ID",
  route_ids: "Routes used by this Demo",
  routes: "Routes",
  worker_ids: "Worker order",
};

function formatValidationErrors(details: unknown[], status: number) {
  const messages = details.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const issue = item as { loc?: unknown; msg?: unknown };
    if (typeof issue.msg !== "string") return [];
    const location = Array.isArray(issue.loc)
      ? issue.loc.filter((part) => part !== "body").map((part) =>
          typeof part === "number" ? `item ${part + 1}` : FIELD_LABELS[String(part)] ?? String(part).replaceAll("_", " ")
        ).join(" → ")
      : "Request";
    return [`${location}: ${issue.msg}`];
  });
  return messages.length
    ? `Validation failed: ${messages.join("; ")}`
    : `Request failed with HTTP ${status}.`;
}
