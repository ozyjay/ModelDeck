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
        : rawDetail && typeof rawDetail === "object" && "message" in rawDetail
          ? String((rawDetail as { message: unknown }).message)
          : `Request failed with HTTP ${response.status}.`;
    throw new Error(detail);
  }
  return payload as T;
}
