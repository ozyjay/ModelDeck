import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, postJson, putJson } from "./api";

describe("API error messages", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("turns HTTP 422 details into readable field locations", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      detail: [{
        type: "string_pattern_mismatch",
        loc: ["body", "routes", 1, "public_name"],
        msg: "String should match the required pattern",
      }],
    }), { status: 422, headers: { "Content-Type": "application/json" } })));

    await expect(putJson("/api/events/example/draft", {})).rejects.toThrow(
      "Validation failed: Routes → item 2 → API Model ID: String should match the required pattern",
    );
  });

  it("keeps structured Event validation details on API errors", async () => {
    const validation = { valid: false, errors: [{ route_id: "route-1", message: "Unknown Worker" }], warnings: [], routes: [] };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      detail: { message: "Event validation failed", validation },
    }), { status: 409, headers: { "Content-Type": "application/json" } })));

    const error = await postJson("/api/events/example/publish").catch((reason) => reason);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 409, detail: { validation } });
  });
});
